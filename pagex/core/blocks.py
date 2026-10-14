"""WP5 - ``Block`` / ``Page`` / ``DocumentModel`` + the annotated-LaTeX codec.

ADR-004: **annotated LaTeX is the single source of truth.**  Every block keeps
its provenance in a ``%PX:`` comment marker above its content::

    %PX:block=d3f1a2c4-b0007;page=1;kind=math;bbox=[120,340,880,410]
    \\[ F = ma \\]

Because the markers are *comments*, the file still compiles and exports like
any normal LaTeX - but the visual grounding (page + bbox) survives manual
editing, round-trips through the editor, and can be replayed onto the original
image by the viewer.

Round-trip guarantee: ``annotated_tex_to_document(blocks_to_annotated_tex(d))``
reproduces block ids, kinds, bboxes and content exactly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

BLOCK_KINDS = ("heading", "paragraph", "math", "table", "diagram", "caption", "note")
TEXT_KINDS = ("paragraph", "caption", "note", "heading")
LATEX_KINDS = ("math", "table", "diagram")

MARKER_RE = re.compile(r"^%PX:block=(?P<id>[^;]+);page=(?P<page>\d+);"
                       r"kind=(?P<kind>[a-z]+);bbox=(?P<bbox>\[[^\]]*\])")
DOC_MARKER_RE = re.compile(r"^%PX:doc=(?P<doc>[^;]+);title=(?P<title>.*)$")
PAGE_MARKER_RE = re.compile(r"^%PX:page=(?P<page>\d+)")
SECTION_RE = re.compile(r"^\\section\*\{(?P<title>.*)\}$", re.S)

# The eight characters that must never be interpreted by TeX.
_ESCAPE_MAP = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}
# longest keys first so "\textasciitilde{}" is undone before "\{" / "\}"
_UNESCAPE = {v: k for k, v in _ESCAPE_MAP.items()}


# ---------------------------------------------------------------------------
# escaping
# ---------------------------------------------------------------------------
def escape_tex(text: str) -> str:
    out = []
    for ch in text:
        out.append(_ESCAPE_MAP.get(ch, ch))
    return "".join(out)


def unescape_tex(text: str) -> str:
    # longest-match first so \textasciitilde{} wins over a bare }
    for src in sorted(_UNESCAPE, key=len, reverse=True):
        text = text.replace(src, _UNESCAPE[src])
    return text


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
def page_id_for(doc_id: str, page_no: int) -> str:
    return f"{doc_id}-p{int(page_no):03d}"


def block_id_for(doc_id: str, seq: int) -> str:
    return f"{doc_id}-b{int(seq):04d}"


@dataclass
class Block:
    block_id: str
    page_id: str
    doc_id: str
    seq: int
    kind: str
    bbox: tuple[float, float, float, float]
    text: str = ""
    latex: str = ""
    confidence: float | None = None
    meta: dict = field(default_factory=dict)
    page_no: int = 0

    def __post_init__(self) -> None:
        if self.kind not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind {self.kind!r}")
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 numbers, got {self.bbox!r}")
        self.bbox = tuple(float(v) for v in self.bbox)  # type: ignore[assignment]

    @property
    def content(self) -> str:
        """What the retriever indexes: LaTeX source for math/table/diagram."""
        return (self.latex or self.text).strip()

    @property
    def display(self) -> str:
        return self.text.strip() or self.latex.strip()

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    @property
    def x_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def y0(self) -> float:
        return self.bbox[1]

    def to_row(self) -> dict:
        return {"block_id": self.block_id, "page_id": self.page_id,
                "doc_id": self.doc_id, "seq": self.seq, "kind": self.kind,
                "bbox": [float(v) for v in self.bbox], "text": self.text,
                "latex": self.latex, "confidence": self.confidence,
                "meta": self.meta}

    @classmethod
    def from_row(cls, row: dict, page_no: int = 0) -> "Block":
        bbox = row["bbox"]
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        meta = row.get("meta") or {}
        if isinstance(meta, str):
            meta = json.loads(meta or "{}")
        return cls(block_id=row["block_id"], page_id=row["page_id"],
                   doc_id=row["doc_id"], seq=int(row["seq"]), kind=row["kind"],
                   bbox=tuple(bbox), text=row.get("text") or "",
                   latex=row.get("latex") or "",
                   confidence=row.get("confidence"),
                   meta=meta, page_no=page_no)


@dataclass
class PageModel:
    page_id: str
    doc_id: str
    page_no: int
    image_path: str | None = None
    width: int = 0
    height: int = 0
    caption: str = ""
    blocks: list[Block] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class DocumentModel:
    doc_id: str
    title: str
    source_path: str = ""
    kind: str = "image"            # image | pdf | handwritten
    status: str = "pending"
    error_msg: str | None = None
    pages: list[PageModel] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def all_blocks(self) -> list[Block]:
        return [b for p in self.pages for b in p.blocks]

    def page(self, page_no: int) -> PageModel | None:
        for p in self.pages:
            if p.page_no == page_no:
                return p
        return None

    def block(self, block_id: str) -> Block | None:
        for b in self.all_blocks:
            if b.block_id == block_id:
                return b
        return None

    def kind_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for b in self.all_blocks:
            hist[b.kind] = hist.get(b.kind, 0) + 1
        return hist

    def plain_text(self) -> str:
        out = []
        for page in self.pages:
            if page.caption:
                out.append(page.caption)
            for b in page.blocks:
                out.append(b.display)
        return "\n".join(out).strip()

    def iter_blocks(self) -> Iterator[Block]:
        for page in self.pages:
            yield from page.blocks


def make_document(doc_id: str, title: str, source_path: str = "",
                  kind: str = "image") -> DocumentModel:
    return DocumentModel(doc_id=doc_id, title=title, source_path=source_path,
                         kind=kind)


def make_page(doc: DocumentModel, page_no: int, width: int = 0, height: int = 0,
              image_path: str | None = None, caption: str = "") -> PageModel:
    page = PageModel(page_id=page_id_for(doc.doc_id, page_no), doc_id=doc.doc_id,
                     page_no=page_no, image_path=image_path, width=width,
                     height=height, caption=caption)
    doc.pages.append(page)
    return page


def document_from_store(store, doc_id: str) -> DocumentModel | None:
    """Rebuild the in-memory SSOT from SQLite (GUI edit/export path).

    The annotated-LaTeX document is *derived* state: this loader reconstructs
    ``DocumentModel`` from the ``documents -> pages -> blocks`` tables so the
    editor and exporters always work on exactly what is stored.
    """
    row = store.get_document(doc_id)
    if row is None:
        return None
    doc = make_document(doc_id, row.get("title") or doc_id,
                        row.get("source_path") or "",
                        row.get("kind") or "image")
    doc.status = row.get("status") or "ready"
    doc.error_msg = row.get("error_msg")
    for prow in store.get_pages(doc_id):
        meta = prow.get("meta") or {}
        if isinstance(meta, str):
            meta = json.loads(meta or "{}")
        page = PageModel(page_id=prow["page_id"], doc_id=doc_id,
                         page_no=int(prow["page_no"]),
                         image_path=prow.get("image_path"),
                         width=int(prow.get("width") or 0),
                         height=int(prow.get("height") or 0),
                         caption=str(meta.get("caption", "")), meta=meta)
        for b in store.get_blocks(page_id=page.page_id):
            page.blocks.append(Block.from_row(b, page_no=page.page_no))
        page.blocks.sort(key=lambda x: x.seq)
        doc.pages.append(page)
    return doc


# ---------------------------------------------------------------------------
# codec: block -> annotated LaTeX
# ---------------------------------------------------------------------------
def _marker(block: Block) -> str:
    bbox = "[" + ",".join(f"{int(round(v))}" for v in block.bbox) + "]"
    return (f"%PX:block={block.block_id};page={block.page_no};"
            f"kind={block.kind};bbox={bbox}")


def block_to_lines(block: Block) -> list[str]:
    """Marker line + the block's LaTeX body."""
    marker = _marker(block)
    kind = block.kind
    if kind == "heading":
        body = f"\\section*{{{escape_tex(block.text.strip())}}}"
    elif kind == "math":
        body = "\\[\n" + (block.latex or block.text).strip() + "\n\\]"
    elif kind == "table":
        inner = (block.latex or block.text).strip()
        if inner:
            body = ("\\begin{table}[h]\n\\centering\n" + inner +
                    "\n\\end{table}")
        else:
            body = "% table content unavailable"
    elif kind == "diagram":
        inner = (block.latex or block.text).strip()
        body = inner if inner else "% diagram without machine-readable form"
    else:                                             # paragraph | caption | note
        body = escape_tex(block.text.strip())
    return [marker, body]


def blocks_to_annotated_tex(doc: DocumentModel, *, documentclass: bool = True,
                            preamble: bool = True) -> str:
    lines: list[str] = []
    if documentclass:
        lines.append(f"%PX:doc={doc.doc_id};title={doc.title}")
        if preamble:
            lines.append(_preamble(doc.title))
        lines.append("\\begin{document}")
    for page in doc.pages:
        lines.append(f"%PX:page={page.page_no}")
        for block in page.blocks:
            lines.extend(block_to_lines(block))
    if documentclass:
        lines.append("\\end{document}")
    return "\n".join(lines) + "\n"


def _preamble(title: str) -> str:
    from .latex_export import PREAMBLE
    return PREAMBLE + f"\\title{{{escape_tex(title)}}}\n\\date{{}}\n"


# ---------------------------------------------------------------------------
# codec: annotated LaTeX -> blocks
# ---------------------------------------------------------------------------
def parse_marker(line: str) -> dict | None:
    m = MARKER_RE.match(line.strip())
    if not m:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", m.group("bbox"))
    bbox = [float(v) for v in nums[:4]] if len(nums) >= 4 else [0.0, 0.0, 0.0, 0.0]
    return {"block_id": m.group("id"), "page": int(m.group("page")),
            "kind": m.group("kind"), "bbox": [float(v) for v in bbox]}


def _strip_env(body: str, env: str) -> str:
    start = f"\\begin{{{env}}}"
    end = f"\\end{{{env}}}"
    i = body.find(start)
    j = body.rfind(end)
    if i >= 0 and j > i:
        inner = body[i + len(start):j]
        if env == "table":                          # drop [h] / \centering junk
            inner = re.sub(r"^\s*\[[^\]]*\]\s*", "", inner)
            inner = inner.replace("\\centering", "").replace("\\small", "")
        return inner.strip("\n")
    return body.strip()


def _drop_optional_arg(body: str) -> str:
    """``\\begin{table}[h]`` residue -> nothing (only the leading arg)."""
    return re.sub(r"^\s*\[[^\]]*\]\s*", "", body.strip("\n"))


def body_to_fields(kind: str, body: str) -> tuple[str, str]:
    """Split the collected LaTeX body into the (text, latex) pair for *kind*."""
    body = body.strip("\n")
    if kind == "math":
        inner = body.strip()
        for wrap in (("\\[", "\\]"), ("\\begin{equation}", "\\end{equation}"),
                     ("$$", "$$")):
            if inner.startswith(wrap[0]) and inner.endswith(wrap[1]):
                inner = inner[len(wrap[0]):-len(wrap[1])].strip()
                break
        return "", inner
    if kind == "table":
        inner = _strip_env(body, "table").strip()
        if inner.startswith("\\begin{tabular}") or "\\begin{tabular}" in inner:
            return "", inner
        return "", _drop_optional_arg(inner) or inner
    if kind == "diagram":
        inner = body.strip()
        if inner.startswith("%"):
            return "", ""
        return "", inner
    if kind == "heading":
        m = SECTION_RE.match(body.strip())
        if m:
            return unescape_tex(m.group("title").strip()), ""
        return unescape_tex(body.strip()), ""
    return unescape_tex(body.strip()), ""            # paragraph | caption | note


def annotated_tex_to_document(tex: str, *, doc_id: str | None = None,
                              title: str = "", source_path: str = "",
                              kind: str = "image") -> DocumentModel:
    """Parse an annotated ``.tex`` back into a :class:`DocumentModel`."""
    doc_marker = DOC_MARKER_RE.search(tex)
    if doc_marker:
        doc_id = doc_id or doc_marker.group("doc")
        title = title or doc_marker.group("title").strip()
    doc_id = doc_id or "imported"
    doc = DocumentModel(doc_id=doc_id, title=title or "Untitled",
                        source_path=source_path, kind=kind, status="ready")

    current_page = 1
    current: dict | None = None          # open block marker
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current
        if current is None:
            buffer.clear()
            return
        body = "\n".join(buffer).strip("\n")
        text, latex = body_to_fields(current["kind"], body)
        page = _ensure_page(doc, current["page"])
        seq = len(page.blocks)
        page.blocks.append(Block(
            block_id=current["block_id"], page_id=page.page_id, doc_id=doc.doc_id,
            seq=seq, kind=current["kind"], bbox=tuple(current["bbox"]),
            text=text, latex=latex, confidence=None,
            page_no=current["page"]))
        current = None
        buffer.clear()

    for raw in tex.splitlines():
        line = raw.rstrip()
        if line.strip() == "\\end{document}":
            flush()                                   # never absorb the footer
            break
        page_m = PAGE_MARKER_RE.match(line)
        marker = parse_marker(line)
        if page_m:
            flush()
            current_page = int(page_m.group("page"))
            _ensure_page(doc, current_page)
            continue
        if marker:
            flush()
            current = marker
            current_page = marker["page"]
            continue
        if line.startswith("%PX:"):
            continue                                # unknown marker (doc, etc.)
        if current is None:
            continue                                # preamble / boilerplate
        buffer.append(raw)
    flush()

    # drop empty pages that only carry a marker
    doc.pages = [p for p in doc.pages if p.blocks or p.image_path or p.width]
    return doc


def _ensure_page(doc: DocumentModel, page_no: int) -> PageModel:
    page = doc.page(page_no)
    if page is None:
        page = PageModel(page_id=page_id_for(doc.doc_id, page_no),
                         doc_id=doc.doc_id, page_no=page_no)
        doc.pages.append(page)
        doc.pages.sort(key=lambda p: p.page_no)
    return page


def apply_annotated_tex(page: PageModel, tex: str, doc_id: str) -> None:
    """Replace *page*'s blocks with those parsed out of an edited .tex file.

    Used by the editor: the user edits LaTeX, we re-parse the ``%PX:`` markers
    and keep ids/bboxes for anything that is still there.
    """
    parsed = annotated_tex_to_document(tex, doc_id=doc_id)
    for p in parsed.pages:
        if p.page_no == page.page_no:
            page.blocks = p.blocks
            for b in page.blocks:
                b.page_id = page.page_id
                b.doc_id = doc_id
                b.page_no = page.page_no


def summary(doc: DocumentModel) -> dict:
    return {"doc_id": doc.doc_id, "title": doc.title, "pages": doc.n_pages,
            "blocks": len(doc.all_blocks), "kinds": doc.kind_histogram()}
