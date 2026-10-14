"""TC-UNIT-06 - annotated-LaTeX codec: escaping + round-trip identity."""
from __future__ import annotations

import pytest

from pagex.core.blocks import (Block, DocumentModel, annotated_tex_to_document,
                               block_id_for, blocks_to_annotated_tex,
                               escape_tex, page_id_for, parse_marker,
                               unescape_tex, make_document, make_page)


def make_doc() -> DocumentModel:
    doc = make_document("d3f1a2c4b5", "Rotational Motion & Kinetics", kind="handwritten")
    page = make_page(doc, 1, width=1240, height=1754, image_path="p1.png")
    blocks = [
        Block(block_id=block_id_for(doc.doc_id, 0), page_id=page.page_id,
              doc_id=doc.doc_id, seq=0, kind="heading", page_no=1,
              bbox=(70.0, 40.0, 1170.0, 120.0), text="Physics - Rotational Motion"),
        Block(block_id=block_id_for(doc.doc_id, 1), page_id=page.page_id,
              doc_id=doc.doc_id, seq=1, kind="paragraph", page_no=1,
              bbox=(70.0, 140.0, 1170.0, 300.0),
              text="Torque equals I times alpha & it's proportional to r x F."),
        Block(block_id=block_id_for(doc.doc_id, 2), page_id=page.page_id,
              doc_id=doc.doc_id, seq=2, kind="math", page_no=1,
              bbox=(80.0, 320.0, 700.0, 420.0),
              latex=r"\tau = I \alpha \quad (a_b = r \times \beta)"),
        Block(block_id=block_id_for(doc.doc_id, 3), page_id=page.page_id,
              doc_id=doc.doc_id, seq=3, kind="table", page_no=1,
              bbox=(80.0, 460.0, 1100.0, 700.0),
              latex=r"\begin{tabular}{ll}" "\n" r"Trial & Energy \\" "\n"
                    r"1 & 0.31 \\" "\n" r"\end{tabular}"),
        Block(block_id=block_id_for(doc.doc_id, 4), page_id=page.page_id,
              doc_id=doc.doc_id, seq=4, kind="diagram", page_no=1,
              bbox=(80.0, 740.0, 900.0, 1000.0),
              latex=r"\draw[->] (0,0) -- (2,0);"),
        Block(block_id=block_id_for(doc.doc_id, 5), page_id=page.page_id,
              doc_id=doc.doc_id, seq=5, kind="note", page_no=1,
              bbox=(70.0, 1040.0, 1170.0, 1140.0),
              text="Remember: 100% of _special_ chars must escape & survive!"),
    ]
    page.blocks.extend(blocks)
    return doc


# ---------------------------------------------------------------------------
# escaping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    "a_b & c$",
    "100% #1 {x} ~ y ^2",
    "cost: $5 & 10% tips",
    "path/to\\file",
])
def test_escape_roundtrip(raw: str):
    assert unescape_tex(escape_tex(raw)) == raw


def test_escape_covers_all_eight_specials():
    for ch in "&%$#_{}":
        escaped = escape_tex(ch)
        assert escaped.startswith("\\")
        assert ch not in escape_tex(ch).replace("\\" + ch, "")


def test_escape_leaves_normal_text_alone():
    assert escape_tex("plain text 123") == "plain text 123"


# ---------------------------------------------------------------------------
# round trip: blocks -> tex -> blocks
# ---------------------------------------------------------------------------
def test_blocks_to_tex_to_blocks_is_identity():
    doc = make_doc()
    tex = blocks_to_annotated_tex(doc)
    parsed = annotated_tex_to_document(tex, doc_id=doc.doc_id, title=doc.title)

    original = doc.all_blocks
    restored = parsed.all_blocks
    assert len(restored) == len(original)
    for a, b in zip(original, restored):
        assert a.block_id == b.block_id, "block ids preserved"
        assert a.kind == b.kind
        assert [round(v) for v in a.bbox] == [round(v) for v in b.bbox], \
            "bbox preserved (grounding survives)"
        assert a.page_no == b.page_no
        assert a.content == b.content, \
            f"content mismatch for {a.block_id}: {a.content!r} != {b.content!r}"


def test_roundtrip_survives_editing_a_paragraph():
    doc = make_doc()
    tex = blocks_to_annotated_tex(doc)
    edited = tex.replace("Torque equals I times alpha", "Torque equals moment of inertia")
    parsed = annotated_tex_to_document(edited, doc_id=doc.doc_id)
    assert "moment of inertia" in parsed.all_blocks[1].text
    # untouched blocks keep their ids/bboxes
    assert parsed.all_blocks[0].block_id == doc.all_blocks[0].block_id
    assert parsed.all_blocks[2].kind == "math"


def test_markers_are_present_and_parseable():
    doc = make_doc()
    tex = blocks_to_annotated_tex(doc)
    markers = [ln for ln in tex.splitlines() if ln.startswith("%PX:block=")]
    assert len(markers) == len(doc.all_blocks)
    first = parse_marker(markers[0])
    assert first["page"] == 1
    assert first["kind"] == "heading"
    assert len(first["bbox"]) == 4
    assert first["bbox"][0] == pytest.approx(70, abs=1)


def test_marker_is_a_comment_so_it_still_compiles():
    doc = make_doc()
    tex = blocks_to_annotated_tex(doc)
    assert tex.startswith("%PX:doc=")
    assert "\\documentclass" in tex
    assert "\\begin{document}" in tex and "\\end{document}" in tex
    for line in tex.splitlines():
        assert not line.startswith("%PX") or line.startswith("%"), \
            "markers must be LaTeX comments"


def test_heading_becomes_section_and_math_becomes_display_math():
    doc = make_doc()
    tex = blocks_to_annotated_tex(doc)
    assert "\\section*{Physics - Rotational Motion}" in tex
    assert "\\[" in tex and "\\tau = I \\alpha" in tex
    assert "\\begin{table}[h]" in tex and "\\begin{tabular}" in tex


def test_page_ids_and_block_ids_are_deterministic():
    assert page_id_for("doc1", 3) == "doc1-p003"
    assert block_id_for("doc1", 7) == "doc1-b0007"


def test_plain_export_strips_markers_and_keeps_content():
    from pagex.core.latex_export import clean_tex
    doc = make_doc()
    clean = clean_tex(blocks_to_annotated_tex(doc))
    assert "%PX:" not in clean
    assert "\\section*{" in clean
    assert "Torque equals I times alpha" in clean, "content survives cleaning"
    # a marker-free file is a public export: no provenance, but still LaTeX
    assert "\\begin{document}" in clean and "\\end{document}" in clean


def test_block_rejects_bad_input():
    with pytest.raises(ValueError):
        Block(block_id="b", page_id="p", doc_id="d", seq=0, kind="nope",
              bbox=(0, 0, 1, 1))
    with pytest.raises(ValueError):
        Block(block_id="b", page_id="p", doc_id="d", seq=0, kind="math",
              bbox=(0, 0, 1))
