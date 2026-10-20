"""WP5 - exporters: clean LaTeX, PDF (pdflatex or ReportLab), DOCX, HTML.

``document_to_tex`` is the annotated SSOT writer; the other three are *views*
over the same :class:`~pagex.core.blocks.DocumentModel` (ADR-004).

PDF strategy (pitfall register): if ``pdflatex`` is on PATH we compile the
real document in draft mode; otherwise ReportLab platypus renders a guaranteed
PDF with **zero TeX installation** - which is what saves the live demo.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .blocks import (Block, DocumentModel, PAGE_MARKER_RE, block_to_lines,
                     blocks_to_annotated_tex, escape_tex, parse_marker,
                     unescape_tex)
from .config import CFG

log = logging.getLogger(__name__)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
"""

EXPORT_FORMATS = ("tex", "clean-tex", "pdf", "docx", "html", "json", "md")
#: ``organized`` = reflowed, typeset document; ``faithful`` = the original
#: page raster with an invisible searchable text layer (layout preserved).
PDF_LAYOUTS = ("organized", "faithful")


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def document_to_tex(doc: DocumentModel, *, annotated: bool = True) -> str:
    """Full compilable .tex (annotated with ``%PX:`` markers or cleaned)."""
    tex = blocks_to_annotated_tex(doc, documentclass=True, preamble=True)
    return tex if annotated else clean_tex(tex)


def clean_tex(tex: str) -> str:
    """Strip every ``%PX:`` provenance line (public export)."""
    return "\n".join(line for line in tex.splitlines()
                     if not line.startswith("%PX:")) .rstrip() + "\n"


def _safe_name(title: str, fallback: str = "document") -> str:
    name = re.sub(r"[^\w\-. ]+", "", title).strip().replace(" ", "_")
    return name[:80] or fallback


def has_pdflatex() -> bool:
    return shutil.which("pdflatex") is not None


def compile_pdf(tex: str, out_path: str | Path, *, timeout: int = 120) -> Path:
    """Compile *tex* with pdflatex (needs a TeX install on the machine)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Windows: never flash a console window when the GUI exports a PDF
    no_window = {"creationflags": subprocess.CREATE_NO_WINDOW} \
        if os.name == "nt" else {}
    with tempfile.TemporaryDirectory(dir=str(out_path.parent)) as tmp:
        src = Path(tmp) / "main.tex"
        src.write_text(tex, encoding="utf-8", newline="\n")
        env = dict(os.environ)
        env.setdefault("max_print_line", "1000")
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-draftmode", "main.tex"],
            cwd=tmp, capture_output=True, text=True, timeout=timeout, env=env,
            **no_window)
        if proc.returncode != 0:                     # retry without -draftmode msg
            raise RuntimeError(f"pdflatex failed:\n{proc.stdout[-2000:]}")
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=tmp, capture_output=True, text=True, timeout=timeout, env=env,
            **no_window)
        produced = Path(tmp) / "main.pdf"
        if proc.returncode != 0 or not produced.exists():
            raise RuntimeError(f"pdflatex failed:\n{proc.stdout[-2000:]}")
        shutil.copyfile(produced, out_path)
    return out_path


# ---------------------------------------------------------------------------
# ReportLab fallback PDF (no TeX required)
# ---------------------------------------------------------------------------
def _rl_paragraph(text: str, style):
    from reportlab.platypus import Paragraph
    return Paragraph(escape_xml(text), style)


def escape_xml(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_RULERS = re.compile(r"\\(hline|toprule|midrule|bottomrule|centering|small|large|"
                     r"begin\{tabular\}\{[^}]*\}|end\{tabular\}|caption\{[^}]*\})")


def _tabular_rows(latex: str) -> list[list[str]]:
    """Shared tabular parser: LaTeX table source -> list of cell rows.

    Used by the DOCX, ReportLab, HTML and Markdown exporters so all four
    views of the same block agree on the cell contents (ADR-004: one source,
    many views).
    """
    if not (latex or "").strip():
        return []
    body = latex
    m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", latex, re.S)
    if m:
        body = m.group(1)
    body = _RULERS.sub(" ", body)
    rows: list[list[str]] = []
    for line in re.split(r"\\\\", body):
        line = line.strip().strip("\\").strip()
        if not line:
            continue
        cells = [_tex_to_plain(c).strip() for c in line.split("&")]
        if any(cells):
            rows.append(cells)
    return rows


def reportlab_pdf(doc: DocumentModel, out_path: str | Path) -> Path:
    """Organized, reflowed PDF (no TeX required).

    Unlike :func:`faithful_pdf`, this view discards the original geometry and
    typesets the document for readability: title, page sections, headings,
    equations, tables and captions in a clean single-column flow.
    """
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("px_body", parent=styles["BodyText"],
                          fontName="Helvetica", fontSize=10, leading=15,
                          spaceAfter=6)
    head = ParagraphStyle("px_head", parent=styles["Heading1"], fontSize=14,
                          spaceBefore=10, spaceAfter=6)
    mono = ParagraphStyle("px_mono", parent=styles["Code"], fontName="Courier",
                          fontSize=9, leading=12)
    cap = ParagraphStyle("px_cap", parent=body, textColor="#64748b",
                         fontSize=8.5, leading=12)
    page_label = ParagraphStyle("px_page", parent=styles["Heading2"],
                                fontSize=11, spaceBefore=0, spaceAfter=4,
                                textColor="#475569")
    title_style = ParagraphStyle("px_title", parent=styles["Title"],
                                 alignment=TA_CENTER, spaceAfter=4)
    subtitle = ParagraphStyle("px_sub", parent=cap, alignment=TA_CENTER,
                              spaceAfter=18)

    story = [Paragraph(escape_xml(doc.title), title_style),
             Paragraph(f"Digitized with PageX · {doc.n_pages} page(s)",
                       subtitle)]
    for i, page in enumerate(doc.pages):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(f"Page {page.page_no}", page_label))
        if page.caption:
            story.append(Paragraph(escape_xml(page.caption), cap))
            story.append(Spacer(1, 4))
        for block in page.blocks:
            _append_rl_block(story, block, body, head, mono)

    def _footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColorRGB(0.45, 0.48, 0.54)
        canvas.drawString(2 * cm, 1.1 * cm, "PageX")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm,
                               f"{canvas.getPageNumber()}")
        canvas.restoreState()

    SimpleDocTemplate(str(out_path), pagesize=A4, title=doc.title,
                      author="PageX", leftMargin=2.2 * cm,
                      rightMargin=2.2 * cm, topMargin=2.0 * cm,
                      bottomMargin=1.8 * cm,
                      onFirstPage=_footer, onLaterPages=_footer).build(story)
    return out_path


def _resolve_image(path: str | None) -> Path | None:
    """Resolve an artifact image path (relative paths use CFG.artifacts_dir)."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = CFG.artifacts_dir / p
    return p if p.exists() else None


def _plain_block_text(block: Block) -> str:
    if block.kind in ("math", "table", "diagram"):
        text = block.latex or block.text or ""
    else:
        text = block.text or block.latex or ""
    return re.sub(r"\s+", " ", text).strip()


def _wrap_for_width(text: str, font: str, size: float, width: float) -> list[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = text.split()
    if not words:
        return []
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if stringWidth(candidate, font, size) <= max(1.0, width):
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def faithful_pdf(doc: DocumentModel, out_path: str | Path) -> Path:
    """Same-layout PDF: original page raster + invisible searchable text layer.

    This is the export for users who chose the VLM precisely because they do
    not want to lose the visual layout.  Every page keeps the source geometry
    (deskewed/upscaled page image) and the VLM blocks are written underneath as
    an invisible text layer, so the PDF stays selectable and searchable.
    """
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pageCompression=1)
    c.setTitle(doc.title)
    c.setAuthor("PageX")
    c.setSubject("Layout-faithful digitization")
    # PageX stores pages at 200 dpi.
    pt_per_px = 72.0 / 200.0
    for page in doc.pages:
        w = float(page.width or 1240) * pt_per_px
        h = float(page.height or 1754) * pt_per_px
        c.setPageSize((w, h))
        image = _resolve_image(page.image_path)
        if image is not None:
            try:
                c.drawImage(ImageReader(str(image)), 0, 0, width=w, height=h,
                            preserveAspectRatio=True, anchor="c", mask="auto")
            except Exception as exc:                # unreadable/corrupt image
                log.warning("faithful PDF: cannot draw %s: %s", image, exc)
                image = None
        if image is None:
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, w, h, stroke=0, fill=1)

        # Invisible but extractable text, positioned at the VLM block bbox.
        # If the page raster is missing, render the same text visibly at the
        # original coordinates so the PDF is still useful.
        visible_fallback = image is None
        for block in page.blocks:
            text = _plain_block_text(block)
            if not text:
                continue
            x0, y0, x1, y1 = [float(v) for v in block.bbox]
            bx = x0 / 1000.0 * w
            by = h - y1 / 1000.0 * h
            bw = max(4.0, (x1 - x0) / 1000.0 * w)
            bh = max(4.0, (y1 - y0) / 1000.0 * h)
            size = max(4.0, min(16.0, bh * 0.55))
            safe = text.encode("latin-1", "replace").decode("latin-1")
            lines = _wrap_for_width(safe, "Helvetica", size, bw)
            max_lines = max(1, int(bh / max(size * 1.15, 1.0)))
            text_obj = c.beginText(bx, by + bh - size)
            try:
                text_obj.setTextRenderMode(0 if visible_fallback else 3)
            except Exception:                       # pragma: no cover
                pass
            text_obj.setFont("Helvetica", size)
            text_obj.setLeading(size * 1.15)
            for line in lines[:max_lines]:
                text_obj.textLine(line)
            c.drawText(text_obj)
        c.showPage()
    c.save()
    return out_path


def _append_rl_block(story, block: Block, body, head, mono) -> None:
    from reportlab.platypus import Paragraph, Preformatted, Spacer
    kind = block.kind
    if kind == "heading":
        story.append(Paragraph(escape_xml(block.text), head))
    elif kind == "math":
        story.append(Preformatted(block.latex or block.text, mono))
        story.append(Spacer(1, 6))
    elif kind == "table":
        story.append(_rl_from_tabular(block.latex or block.text, mono))
        story.append(Spacer(1, 6))
    elif kind == "diagram":
        story.append(Preformatted(block.latex or block.text, mono))
        story.append(Spacer(1, 6))
    else:
        if block.text.strip():
            story.append(Paragraph(escape_xml(block.text), body))
            story.append(Spacer(1, 6))


def _rl_from_tabular(latex: str, mono):
    """Minimal ``\\begin{tabular}{...}`` -> ReportLab table (TeX-less PDF path)."""
    from reportlab.platypus import Preformatted, Table, TableStyle
    from reportlab.lib import colors
    rows = [[_tex_to_plain(c) for c in row] for row in _tabular_rows(latex)]
    if not rows:
        return Preformatted(latex, mono)
    cols = max(len(r) for r in rows)
    rows = [r + [""] * (cols - len(r)) for r in rows]
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(.9, .9, .9)),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _tex_to_plain(text: str) -> str:
    text = re.sub(r"\\(textbf|textit|emph|mathrm|mathbf)\{([^}]*)\}", r"\2", text)
    text = text.replace("\\", "")
    return unescape_tex(text.strip())


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
def export_docx(doc: DocumentModel, out_path: str | Path) -> Path:
    from docx import Document
    from docx.shared import Pt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = Document()
    d.add_heading(doc.title, level=0)
    for page in doc.pages:
        for block in page.blocks:
            kind = block.kind
            if kind == "heading":
                d.add_heading(block.text, level=1)
            elif kind == "math":
                p = d.add_paragraph()
                run = p.add_run(block.latex or block.text)
                run.font.name = "Cambria Math"
                run.italic = True
            elif kind == "table":
                _docx_table(d, block)
            elif kind == "diagram":
                p = d.add_paragraph()
                run = p.add_run("[diagram] " + (block.latex or block.text))
                run.font.size = Pt(8)
                run.font.name = "Consolas"
            else:
                if block.text.strip():
                    d.add_paragraph(block.text)
        if page.caption:
            d.add_paragraph(page.caption).italic = True
    d.save(str(out_path))
    return out_path


def _docx_table(d, block: Block) -> None:
    rows = _tabular_rows(block.latex or "")
    if not rows:
        d.add_paragraph(block.latex or "")
        return
    cols = max(len(r) for r in rows)
    table = d.add_table(rows=len(rows), cols=cols)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j in range(cols):
            table.rows[i].cells[j].text = row[j] if j < len(row) else ""


# ---------------------------------------------------------------------------
# HTML (instant editor preview - no TeX, no Chromium)
# ---------------------------------------------------------------------------
_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
     max-width:52rem;margin:2rem auto;padding:0 1rem;color:#16161d;line-height:1.6}
h1{font-size:1.7rem;border-bottom:2px solid #eee;padding-bottom:.4rem}
h2{font-size:1.25rem;margin-top:1.6rem;color:#2b2b3a}
.math{background:#f7f7fb;border-left:3px solid #7c6cf0;padding:.7rem 1rem;
      font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;
      overflow-x:auto;margin:.8rem 0}
table{border-collapse:collapse;margin:.8rem 0;font-size:.92rem}
td,th{border:1px solid #d6d6e2;padding:.35rem .6rem}
.cap{color:#6b6b7b;font-size:.88rem;font-style:italic}
.px{color:#c0c0cc;font-size:.75rem;font-family:monospace}
.img{max-width:100%;border:1px solid #e3e3ee;border-radius:6px;margin:.6rem 0}
"""


def document_to_html(doc: DocumentModel, *, show_markers: bool = False,
                     page_image_urls: dict[str, str] | None = None) -> str:
    """Self-contained HTML preview for the editor's right pane."""
    page_image_urls = page_image_urls or {}
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>{escape_xml(doc.title)}</title><style>{_CSS}</style></head>",
             "<body>", f"<h1>{escape_xml(doc.title)}</h1>"]
    for page in doc.pages:
        url = page_image_urls.get(page.page_id) or page.image_path
        if url:
            parts.append(f"<img class='img' src='{escape_xml(str(url))}' "
                         f"alt='page {page.page_no}'>")
        if page.caption:
            parts.append(f"<p class='cap'>{escape_xml(page.caption)}</p>")
        for block in page.blocks:
            if show_markers:
                parts.append(f"<div class='px'>%PX:block={block.block_id};"
                             f"page={block.page_no};kind={block.kind}</div>")
            kind = block.kind
            if kind == "heading":
                parts.append(f"<h2>{escape_xml(block.text)}</h2>")
            elif kind == "math":
                parts.append(f"<div class='math'>{escape_xml(block.latex or block.text)}</div>")
            elif kind == "table":
                parts.append(_html_table(block.latex or block.text))
            elif kind == "diagram":
                inner = (block.latex or "").strip()
                body = escape_xml(inner) if inner else "(diagram)"
                parts.append(f"<div class='math'>{body}</div>")
            elif block.text.strip():
                parts.append(f"<p>{escape_xml(block.text)}</p>")
        parts.append(f"<p class='px'>— page {page.page_no} —</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _html_table(latex: str) -> str:
    rows = _tabular_rows(latex)
    if not rows:
        return f"<div class='math'>{escape_xml(latex)}</div>"
    cols = max(len(r) for r in rows)
    html = ["<table>"]
    for i, row in enumerate(rows):
        tag = "th" if i == 0 else "td"
        row = row + [""] * (cols - len(row))
        html.append("<tr>" + "".join(f"<{tag}>{escape_xml(c)}</{tag}>"
                                     for c in row) + "</tr>")
    html.append("</table>")
    return "".join(html)


def latex_to_html_fragment(tex: str) -> str:
    """Cheap TeX -> HTML for raw source in the editor (not a full renderer)."""
    frag = clean_tex(tex)
    frag = re.sub(r"\\section\*\{([^}]*)\}", r"<h2>\1</h2>", frag)
    frag = re.sub(r"\\\[|\\\]|\\begin\{equation\}|\\end\{equation\}", "", frag)
    frag = re.sub(r"\\(textbf|emph|textit)\{([^}]*)\}", r"<b>\2</b>", frag)
    frag = re.sub(r"^%.*$", "", frag, flags=re.M)
    return f"<pre style='white-space:pre-wrap'>{escape_xml(frag)}</pre>"


def document_to_markdown(doc: DocumentModel) -> str:
    lines = [f"# {doc.title}", ""]
    for page in doc.pages:
        for block in page.blocks:
            kind = block.kind
            if kind == "heading":
                lines += [f"## {block.text}", ""]
            elif kind == "math":
                lines += ["$$", block.latex or block.text, "$$", ""]
            elif kind == "table":
                lines += [_tabular_to_md(block.latex or block.text), ""]
            elif kind == "diagram":
                lines += ["```tikz", block.latex or "", "```", ""]
            elif block.text.strip():
                lines += [block.text, ""]
        if page.caption:
            lines += [f"> {page.caption}", ""]
    return "\n".join(lines).strip() + "\n"


def _tabular_to_md(latex: str) -> str:
    rows = _tabular_rows(latex)
    if not rows:
        return "```\n" + latex + "\n```"
    width = max(len(r) for r in rows)
    out = ["| " + " | ".join(rows[0] + [""] * (width - len(rows[0]))) + " |",
           "|" + "---|" * width]
    for row in rows[1:]:
        row = row + [""] * (width - len(row))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# one-call export
# ---------------------------------------------------------------------------
def export_document(doc: DocumentModel, out_path: str | Path,
                    fmt: str, *, pdf_layout: str = "organized") -> Path:
    """Write *fmt* (tex|clean-tex|pdf|docx|html|json|md) to *out_path*.

    For PDF, ``pdf_layout`` selects:

    * ``organized`` - reflowed/typeset document (pdflatex if available, else
      the ReportLab fallback).
    * ``faithful`` - original page image + invisible text layer, preserving the
      input layout exactly.
    """
    from .blocks import summary
    import json as _json

    out_path = Path(out_path)
    fmt = fmt.lower()
    layout = (pdf_layout or "organized").lower()
    if layout not in PDF_LAYOUTS:
        raise ValueError(f"unknown PDF layout {pdf_layout!r} "
                         f"(choose from {PDF_LAYOUTS})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "tex":
        out_path.write_text(document_to_tex(doc, annotated=True),
                            encoding="utf-8", newline="\n")
    elif fmt in ("clean-tex", "clean", "latex"):
        out_path.write_text(document_to_tex(doc, annotated=False),
                            encoding="utf-8", newline="\n")
    elif fmt in ("pdf", "pdf-faithful", "pdf-organized"):
        if fmt == "pdf-faithful" or layout == "faithful":
            return faithful_pdf(doc, out_path)
        if has_pdflatex():
            try:
                return compile_pdf(document_to_tex(doc, annotated=False), out_path)
            except Exception as exc:                 # pragma: no cover
                log.warning("pdflatex failed (%s); ReportLab fallback", exc)
        return reportlab_pdf(doc, out_path)
    elif fmt == "docx":
        return export_docx(doc, out_path)
    elif fmt == "html":
        out_path.write_text(document_to_html(doc), encoding="utf-8", newline="\n")
    elif fmt == "md":
        out_path.write_text(document_to_markdown(doc), encoding="utf-8", newline="\n")
    elif fmt == "json":
        payload = {**summary(doc),
                   "pages": [{"page_no": p.page_no, "image_path": p.image_path,
                              "width": p.width, "height": p.height,
                              "caption": p.caption,
                              "blocks": [b.to_row() for b in p.blocks]}
                             for p in doc.pages]}
        out_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"unknown format {fmt!r} (choose from {EXPORT_FORMATS})")
    return out_path
