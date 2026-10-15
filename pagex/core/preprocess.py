"""WP2 - image preprocessing: zero AI, <100 ms per page.

load (PyMuPDF @200 DPI for PDFs, Pillow for photos) -> upscale small pages ->
grayscale -> CLAHE -> deskew -> save page PNG + 256 px thumbnail.

Every function is pure (ndarray in / ndarray out) so the unit tests can feed
synthetic fixtures without touching a model.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PDF_DPI = 200
MIN_WIDTH = 1500          # handwriting strokes need pixels (ADR: upscale x2)
SKEW_EPS_DEG = 0.5        # below this we call the page straight
THUMB_PX = 256


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_pdf_page(pdf_path: str | Path, page_no: int, dpi: int = PDF_DPI) -> np.ndarray:
    """Rasterise one PDF page to an RGB ndarray (PyMuPDF = no poppler dep)."""
    import fitz
    doc = fitz.open(str(pdf_path))
    try:
        if not 0 <= page_no < doc.page_count:
            raise IndexError(f"page {page_no} out of range ({doc.page_count})")
        page = doc.load_page(page_no)
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:                       # RGBA -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(img)
    finally:
        doc.close()


def pdf_page_count(pdf_path: str | Path) -> int:
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as RGB uint8 (EXIF orientation respected)."""
    from PIL import ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im) if hasattr(im, "getexif") else im
        im = im.convert("RGB")
        return np.ascontiguousarray(np.array(im))


def load_pages(path: str | Path, dpi: int = PDF_DPI,
               max_pages: int | None = None) -> list[np.ndarray]:
    """All pages of a PDF (capped at *max_pages*), or a single image, as RGB arrays.

    The cap matters for big scans (the bundled AI notes are 126 pages): an
    8 GB laptop must be able to index a document incrementally instead of
    rasterising everything at once.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        n = pdf_page_count(path)
        if max_pages is not None:
            n = min(n, max(1, int(max_pages)))
        return [load_pdf_page(path, i, dpi=dpi) for i in range(n)]
    return [load_image(path)]


# ---------------------------------------------------------------------------
# enhancement
# ---------------------------------------------------------------------------
def ensure_min_width(img: np.ndarray, min_width: int = MIN_WIDTH) -> tuple[np.ndarray, float]:
    """2x Lanczos upscale for low-res phone photos (spec: <1500 px)."""
    h, w = img.shape[:2]
    if w >= min_width:
        return img, 1.0
    scale = 2.0
    interp = cv2.INTER_LANCZOS4
    out = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)
    return out, scale


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def enhance_gray(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE - rescues pale pencil strokes and uneven phone-photo lighting."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def morph_clean(gray: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Close 1-px gaps in pencil strokes (pitfall register).

    Strokes are *dark on bright paper*, so the gap we want to fill is a bright
    speck inside a dark line: invert -> close -> invert.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    inverted = 255 - gray
    filled = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, k)
    return 255 - filled


def _ink_mask(gray: np.ndarray, scale: int = 4) -> np.ndarray | None:
    """Small binary ink mask (1 = ink) for the skew search."""
    h, w = gray.shape[:2]
    sw, sh = max(64, w // scale), max(64, h // scale)
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    thr = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = (thr > 0).astype(np.uint8)
    if mask.sum() < 100:                       # blank / near-blank page
        return None
    return mask


def _rotate_mask(mask: np.ndarray, degrees_: float) -> np.ndarray:
    h, w = mask.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees_, 1.0)
    return cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST,
                          borderValue=0)


def _projection_variance(mask: np.ndarray) -> float:
    """Variance of the horizontal projection profile.

    Level text packs ink into a few dense rows (big variance); a skewed page
    smears every glyph across many rows (small variance).  Maximising this
    score is the classic document-deskew estimator and - unlike ``minAreaRect``
    or a PCA over pixel coordinates - it is independent of the page aspect
    ratio and of which quadrant OpenCV reports.
    """
    profile = mask.sum(axis=1, dtype=np.float64)
    return float(profile.var())


def estimate_skew(gray: np.ndarray, max_angle: float = 12.0,
                  coarse: float = 1.0, fine: float = 0.25) -> float:
    """Angle (deg) to rotate the page by so its text lines run horizontally.

    Coarse-to-fine projection-profile search over ``+-max_angle`` (scan skew is
    almost never worse than 12 deg).  ``0.0`` is always the baseline, so a
    straight page reports exactly 0 - which is what makes deskew a no-op for
    clean input.
    """
    mask = _ink_mask(gray)
    if mask is None:
        return 0.0
    baseline = _projection_variance(mask)
    best_score, best_angle = baseline, 0.0

    def scan(lo: float, hi: float, step: float) -> None:
        nonlocal best_score, best_angle
        a = lo
        while a <= hi + 1e-9:
            if a != 0.0:
                score = _projection_variance(_rotate_mask(mask, a))
                if score > best_score * 1.0005:      # hysteresis: ignore noise
                    best_score, best_angle = score, a
            a += step

    scan(-abs(max_angle), abs(max_angle), coarse)
    scan(best_angle - coarse, best_angle + coarse, fine)
    if abs(best_angle) > abs(max_angle) + 1e-6:
        return 0.0
    return float(round(best_angle, 2))


def deskew(img: np.ndarray, angle: float | None = None) -> tuple[np.ndarray, float]:
    """Rotate so the text lines are horizontal. Returns (image, applied angle).

    The correction is *verified*: we measure the projection score before and
    after, and if rotating made it worse we flip the sign and retry.  A sign
    error therefore cannot survive - and the caller always receives the angle
    that was actually applied, which is what keeps the colour plane in sync.
    """
    gray = img if img.ndim == 2 else to_gray(img)
    if angle is None:
        angle = estimate_skew(gray)
    if abs(angle) < SKEW_EPS_DEG:
        return img, 0.0

    def _rotate(source, degrees_):
        h, w = source.shape[:2]
        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, degrees_, 1.0)
        cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        mat[0, 2] += (nw / 2.0) - center[0]
        mat[1, 2] += (nh / 2.0) - center[1]
        border = (255, 255, 255) if source.ndim == 3 else 255
        return cv2.warpAffine(source, mat, (nw, nh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    def _score(img_) -> float:
        g = img_ if img_.ndim == 2 else to_gray(img_)
        mask = _ink_mask(g)
        return 0.0 if mask is None else _projection_variance(mask)

    before = _score(gray)
    candidate = _rotate(img, angle)
    after = _score(candidate)
    if after < before:                             # wrong way - flip and retry
        flipped = _rotate(img, -angle)
        if _score(flipped) >= after:
            return flipped, float(-angle)
    return candidate, float(angle)


def preprocess_page(img: np.ndarray, *, do_deskew: bool = True) -> dict:
    """Full WP2 chain. Returns a dict of every intermediate product.

    Pure function: no files, no models - which is exactly why TC-UNIT-01/02
    can assert on it with synthetic fixtures.
    """
    original = img
    img, scale = ensure_min_width(img)
    gray = to_gray(img)
    enhanced = enhance_gray(gray)
    enhanced = morph_clean(enhanced)
    angle = 0.0
    if do_deskew:
        enhanced, angle = deskew(enhanced)
        if scale != 1.0 or angle:
            # deskew ran on the gray plane; mirror the geometry on the colour one
            h, w = img.shape[:2]
            if angle:
                mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
                nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
                mat[0, 2] += (nw / 2.0) - (w / 2.0)
                mat[1, 2] += (nh / 2.0) - (h / 2.0)
                img = cv2.warpAffine(img, mat, (nw, nh), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
    return {
        "image": np.ascontiguousarray(img),          # colour, processed
        "gray": np.ascontiguousarray(enhanced),      # mono, processed
        "scale": scale,
        "angle": angle,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "contrast": float(enhanced.std()),
        "orig_width": int(original.shape[1]),
        "orig_height": int(original.shape[0]),
    }


# ---------------------------------------------------------------------------
# side effects (kept apart from the pure functions above)
# ---------------------------------------------------------------------------
def save_page(img: np.ndarray, out_path: str | Path) -> Path:
    """Persist a page array exactly as produced (RGB, or single-channel grey).

    PageX's array convention is **RGB everywhere** (Pillow and PyMuPDF both
    deliver RGB), so PIL must *not* be handed channel-reversed data: reversing
    an RGB array writes a BGR file, which is what made saved pages render with
    swapped red/blue (blue margin lines, orange "remember" lines) in the GUI
    and in every exported page image.  Regression test:
    ``tests/unit/test_preprocess.py::test_save_page_preserves_colour``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    return out_path


def thumbnail(img: np.ndarray, px: int = THUMB_PX) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= px and w <= px:
        return img
    if w >= h:
        size = (px, max(1, int(h * px / w)))
    else:
        size = (max(1, int(w * px / h)), px)
    interp = cv2.INTER_AREA if min(h, w) > max(size) else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)


def rotate_image(img: np.ndarray, degrees: float) -> np.ndarray:
    """Test helper - synthesize a skewed page."""
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def crop_region(img: np.ndarray, bbox: Sequence[float], pad: float = 0.03) -> np.ndarray:
    """Cut a block out of the page with a small relative padding.

    Used by the per-crop fallback: the VLM sees a legible close-up instead of
    a whole 1280 px page with 8 pt handwriting.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, x1 = max(0.0, x0 - px), min(float(w), x1 + px)
    y0, y1 = max(0.0, y0 - py), min(float(h), y1 + py)
    if x1 <= x0 or y1 <= y0:
        return img
    return np.ascontiguousarray(img[int(y0):int(y1), int(x0):int(x1)])


def residual_skew(img: np.ndarray) -> float:
    """|skew| still present after preprocessing (used by TC-UNIT-01)."""
    gray = img if img.ndim == 2 else to_gray(img)
    return abs(estimate_skew(gray))


def ink_ratio(gray: np.ndarray) -> float:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return float((thr > 0).mean())
