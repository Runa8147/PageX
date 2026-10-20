"""TC-UNIT-01 / TC-UNIT-02 - preprocessing (pure functions, no models)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pagex.core import preprocess as pp
from pagex.testing import fixtures


# ---------------------------------------------------------------------------
# TC-UNIT-01: deskew
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [7.0, -7.0, -5.0, 3.5])
def test_deskew_removes_synthetic_rotation(angle):
    """A page rotated by N degrees must come back within 0.5 deg of level."""
    page = np.array(fixtures.handwritten_page(size=(900, 1200), seed=4))
    skewed = pp.rotate_image(page, angle)
    before = pp.residual_skew(skewed)
    fixed, applied = pp.deskew(pp.to_gray(skewed))
    after = pp.residual_skew(fixed if fixed.ndim == 2 else pp.to_gray(fixed))

    assert abs(before - abs(angle)) < 1.5, "fixture should actually be skewed"
    assert after < 0.5, f"residual skew {after:.2f} deg after deskew"
    assert abs(abs(applied) - abs(angle)) < 1.5, \
        f"applied {applied:.2f} for {angle}"


def test_deskew_leaves_straight_page_alone():
    page = np.array(fixtures.handwritten_page(size=(700, 900), seed=9))
    out, applied = pp.deskew(pp.to_gray(page))
    assert applied == 0.0, "already-straight page must not be rotated"


def test_estimate_skew_on_blank_is_zero():
    blank = np.full((400, 600), 255, dtype=np.uint8)
    assert pp.estimate_skew(blank) == 0.0


# ---------------------------------------------------------------------------
# TC-UNIT-02: upscale + CLAHE
# ---------------------------------------------------------------------------
def test_small_note_is_upscaled_2x():
    note = np.array(fixtures.small_note(size=(700, 990)))
    out, scale = pp.ensure_min_width(note)
    assert scale == 2.0
    assert out.shape[1] >= 1400, f"width {out.shape[1]} < 1400"


def test_large_page_is_not_upscaled():
    page = np.array(fixtures.handwritten_page(size=(1600, 2200)))
    out, scale = pp.ensure_min_width(page)
    assert scale == 1.0
    assert out.shape == page.shape


def test_clahe_raises_contrast_on_pale_pencil():
    """Pale strokes (low global contrast) must gain std-dev after CLAHE."""
    pale = np.array(fixtures.small_note(size=(700, 990)))
    gray = pp.to_gray(pale)
    faded = (gray.astype(np.float32) * 0.35 + 0.6 * 255).astype(np.uint8)
    enhanced = pp.enhance_gray(faded)
    assert enhanced.std() > faded.std()


def test_morph_close_fills_stroke_gaps():
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[40:60, 30:70] = 0
    img[48:52, 48:52] = 255            # small hole in the middle of the stroke
    closed = pp.morph_clean(img, kernel=5)
    assert (closed[50, 50] == 0), "morphological close should fill the gap"


# ---------------------------------------------------------------------------
# pipeline helpers used by WP4
# ---------------------------------------------------------------------------
def test_preprocess_page_returns_full_product_set():
    page = np.array(fixtures.small_note())
    out = pp.preprocess_page(page)
    for key in ("image", "gray", "scale", "angle", "width", "height",
                "contrast"):
        assert key in out
    assert out["scale"] == 2.0
    assert out["width"] >= 1400


def test_crop_region_pads_and_stays_inside():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    crop = pp.crop_region(img, (100, 100, 200, 160), pad=0.03)
    assert crop.shape[0] > 60 and crop.shape[1] > 100
    # a bbox flush with the corner must not read outside the image
    corner = pp.crop_region(img, (0, 0, 50, 50), pad=0.1)
    assert corner.shape[0] <= 56 and corner.shape[1] <= 56


def test_thumbnail_is_bounded():
    page = np.array(fixtures.handwritten_page(size=(1240, 1754)))
    thumb = pp.thumbnail(page, px=256)
    assert max(thumb.shape[:2]) == 256


# ---------------------------------------------------------------------------
# colour fidelity (regression: saved pages once came back R/B swapped)
# ---------------------------------------------------------------------------
def test_save_page_preserves_colour(tmp_path):
    """A red page must still be red after ``save_page`` (RGB in -> RGB file).

    ``save_page`` used to hand PIL a channel-reversed array, so every page
    written to the artifacts dir (and shown in the GUI viewer) had its reds
    and blues swapped - blue margin lines, orange highlights.
    """
    red = np.zeros((32, 48, 3), dtype=np.uint8)
    red[..., 0] = 255
    out = pp.save_page(red, tmp_path / "page.png")
    back = np.array(Image.open(out))
    assert back.shape == red.shape
    assert tuple(int(v) for v in back[0, 0]) == (255, 0, 0), \
        "save_page must not swap the red and blue channels"


def test_save_page_roundtrip_is_lossless(tmp_path):
    page = np.array(fixtures.handwritten_page(seed=3).convert("RGB"))
    out = pp.save_page(page, tmp_path / "p.png")
    back = pp.load_image(out)                      # PIL -> RGB
    assert back.shape == page.shape
    assert np.array_equal(back, page), \
        "PNG is lossless: save + load must reproduce the exact pixels"


def test_save_page_writes_grey_pages_as_grey(tmp_path):
    grey = np.full((20, 30), 17, dtype=np.uint8)
    out = pp.save_page(grey, tmp_path / "g.png")
    assert np.array(Image.open(out)).dtype == np.uint8
    assert Image.open(out).mode in ("L", "LA")


# ---------------------------------------------------------------------------
# streaming PDFs (large scans must not be materialised at once)
# ---------------------------------------------------------------------------
def test_iter_pages_streams_and_caps_a_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "three_pages.pdf"
    pdf = fitz.open()
    for i in range(3):
        page = pdf.new_page(width=300, height=400)
        page.insert_text((40, 60), f"page {i + 1}")
    pdf.save(str(path))
    pdf.close()

    assert pp.page_count(path) == 3
    assert pp.page_count(path, max_pages=2) == 2
    pages = pp.iter_pages(path, dpi=72, max_pages=2)
    # a generator, not an eager list
    assert not isinstance(pages, list)
    first = next(pages)
    assert first.ndim == 3 and first.shape[2] == 3
    assert len(list(pages)) == 1
    assert len(pp.load_pages(path, dpi=72, max_pages=2)) == 2
