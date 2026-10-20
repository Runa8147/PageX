"""TC-UNIT-04 - page parsing: JSON extraction, NMS, reading order, rescale."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pagex.core.blocks import BLOCK_KINDS
from pagex.core.parse import (Region, extract_json, iou, nms, parse_page_json,
                              reading_order, regions_to_blocks, rescale,
                              parse_with_crops)
from pagex.testing import fixtures

W, H = 1240, 1754


# ---------------------------------------------------------------------------
# JSON extraction (the "model returns prose" pitfall)
# ---------------------------------------------------------------------------
def test_extract_json_from_prose():
    raw = 'Sure! Here is the JSON you asked for:\n{"caption":"x","regions":[]}\nHope that helps.'
    data = extract_json(raw)
    assert data["caption"] == "x"


def test_extract_json_from_fenced_block():
    raw = '```json\n{"caption":"fenced","regions":[]}\n```'
    assert extract_json(raw)["caption"] == "fenced"


def test_extract_json_nested_strings_with_braces():
    raw = 'noise {"caption":"curly {brace} inside","regions":[]}'
    data = extract_json(raw)
    assert data["caption"] == "curly {brace} inside"


def test_extract_json_raises_on_empty():
    with pytest.raises(ValueError):
        extract_json("")


def test_extract_json_from_bare_array():
    assert extract_json('[{"bbox":[1,2,3,4]}]')[0]["bbox"] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_iou_identical_is_one():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_nms_removes_duplicate_detection():
    """Two boxes with IoU > 0.5 collapse to one (spec: canned JSON, 2 dups)."""
    regions = [
        Region(bbox=(100.0, 100.0, 500.0, 300.0), kind="paragraph", order=0),
        Region(bbox=(110.0, 110.0, 510.0, 310.0), kind="paragraph", order=1),
        Region(bbox=(100.0, 900.0, 500.0, 1100.0), kind="math", order=2),
    ]
    kept = nms(regions, threshold=0.5)
    assert len(kept) == 2
    assert {k.kind for k in kept} == {"paragraph", "math"}


def test_nms_keeps_separate_blocks():
    regions = [
        Region(bbox=(0.0, 0.0, 300.0, 200.0), kind="heading", order=0),
        Region(bbox=(0.0, 400.0, 300.0, 800.0), kind="paragraph", order=1),
    ]
    assert len(nms(regions, threshold=0.5)) == 2


# ---------------------------------------------------------------------------
# reading order
# ---------------------------------------------------------------------------
def test_two_column_page_orders_col_major():
    """Left column entirely before right column, y ascending inside each."""
    regions = [
        Region(bbox=(60.0, 60.0, 1180.0, 130.0), kind="heading"),       # full width
        Region(bbox=(60.0, 150.0, 580.0, 350.0), kind="paragraph"),     # L1
        Region(bbox=(60.0, 500.0, 580.0, 700.0), kind="paragraph"),     # L2
        Region(bbox=(600.0, 150.0, 1180.0, 350.0), kind="paragraph"),   # R1
        Region(bbox=(600.0, 500.0, 1180.0, 700.0), kind="paragraph"),   # R2
    ]
    ordered = reading_order(regions, page_width=W)
    kinds = [r.kind for r in ordered]
    assert kinds[0] == "heading", "full-width title first"
    ys = [(r.bbox[0], r.bbox[1]) for r in ordered[1:]]
    assert ys == sorted(ys, key=lambda t: (0 if t[0] < 300 else 1, t[1])), \
        "left column before right column, ascending within column"


def test_single_column_orders_by_y():
    regions = [Region(bbox=(50.0, y, 1100.0, y + 100), kind="paragraph")
               for y in (600, 200, 400)]
    ordered = reading_order(regions, page_width=W)
    assert [r.bbox[1] for r in ordered] == [200, 400, 600]


def test_reading_order_assigns_consecutive_orders():
    regions = [Region(bbox=(float(x), 100.0, float(x) + 300, 300), kind="paragraph")
               for x in (50, 400, 750)]
    ordered = reading_order(regions, page_width=W)
    assert [r.order for r in ordered] == [0, 1, 2]


# ---------------------------------------------------------------------------
# rescale + full page parse
# ---------------------------------------------------------------------------
def test_rescale_maps_0_1000_to_pixels():
    x0, y0, x1, y1 = rescale([0, 0, 1000, 1000], W, H)
    assert (x0, y0) == (0, 0)
    assert x1 == pytest.approx(W) and y1 == pytest.approx(H)


def test_rescale_clamps_and_swaps_inverted_corners():
    x0, y0, x1, y1 = rescale([800, 900, 200, 100], W, H)
    assert x0 < x1 and y0 < y1
    assert x1 <= W and y1 <= H
    assert rescale([0, 0, 5000, 5000], W, H)[2] == pytest.approx(W)


def test_parse_page_json_keeps_layout_hint():
    payload = ('{"caption":"two columns","layout":"two_column",'
               '"regions":[]}')
    parsed = parse_page_json(payload, W, H)
    assert parsed.caption == "two columns"
    assert parsed.layout == "two_column"


def test_parse_page_json_from_canned_fixture():
    parsed = parse_page_json(fixtures.CANNED_PAGE_JSON, W, H)
    assert parsed.caption
    assert parsed.regions, "regions parsed"
    assert all(r.kind in BLOCK_KINDS for r in parsed.regions)
    # all bboxes are pixel space, inside the page
    for r in parsed.regions:
        assert 0 <= r.bbox[0] < r.bbox[2] <= W
        assert 0 <= r.bbox[1] < r.bbox[3] <= H
    # reading order was re-derived (consecutive, from 0)
    assert [r.order for r in parsed.regions] == list(range(len(parsed.regions)))


def test_parse_drops_tiny_boxes():
    payload = json.dumps({"caption": "tiny", "regions": [
        {"bbox": [10, 10, 30, 30], "kind": "paragraph", "content": "x"},
        {"bbox": [100, 400, 900, 700], "kind": "paragraph", "content": "y"}]})
    parsed = parse_page_json(payload, W, H, min_box_frac=0.01)
    assert len(parsed.regions) == 1
    assert parsed.regions[0].content == "y"


def test_parse_blank_page_is_ok_not_error():
    parsed = parse_page_json('{"caption":"","regions":[]}', W, H)
    assert parsed.regions == []
    assert parsed.ok


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_page_json("I cannot read this page, sorry.", W, H)


def test_unknown_kind_falls_back_to_paragraph():
    payload = json.dumps({"caption": "c", "regions": [
        {"bbox": [100, 100, 900, 400], "kind": "gibberish", "content": "x"}]})
    parsed = parse_page_json(payload, W, H)
    assert parsed.regions[0].kind == "paragraph"
    assert parsed.warnings


# ---------------------------------------------------------------------------
# fallback ladder
# ---------------------------------------------------------------------------
def test_parse_with_crops_transcribes_each_region():
    page = np.array(fixtures.handwritten_page(size=(W, H), seed=7))

    def crop_fn(crop, prompt):
        assert crop is not None and crop.size > 0, "crop must be cut from page"
        return fixtures.CANNED_CROP_JSON
    parsed = parse_with_crops(page, fixtures.CANNED_LAYOUT_JSON, W, H, crop_fn)
    assert parsed.mode == "crops"
    assert parsed.regions
    assert any(r.content for r in parsed.regions)


# ---------------------------------------------------------------------------
# regions -> blocks
# ---------------------------------------------------------------------------
def test_regions_to_blocks_kinds_route_text_vs_latex():
    parsed = parse_page_json(fixtures.CANNED_PAGE_JSON, W, H)
    blocks = regions_to_blocks(parsed, "doc123", page_no=1, seq_offset=0)
    assert blocks
    math_blocks = [b for b in blocks if b.kind == "math"]
    text_blocks = [b for b in blocks if b.kind in ("paragraph", "note", "heading")]
    assert math_blocks and all(b.latex and not b.text for b in math_blocks)
    assert text_blocks and all(b.text and not b.latex for b in text_blocks)
    assert [b.seq for b in blocks] == list(range(len(blocks)))
    assert all(b.doc_id == "doc123" for b in blocks)
