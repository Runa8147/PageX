"""WP4 - page parsing: VLM text -> validated blocks with real pixel bboxes.

One VLM call per page (ADR-008) returns STRICT JSON::

    {"caption": "...",
     "regions":[{"bbox":[x0,y0,x1,y1]  # normalised 0..1000
                 "kind":"heading|paragraph|math|table|diagram|caption|note",
                 "content":"...",
                 "order": 0}]}

The model's ``order`` field is *advisory only*: reading order is re-derived
here (column-aware) so output is deterministic regardless of what the model
felt like that day.  Bboxes are rescaled to page pixels, de-duplicated with
NMS, and tiny boxes are dropped.

Fallback ladder (pitfall register: "VLM returns prose, not JSON"):
  1. strict JSON extract -> retry once with a "JSON ONLY" suffix
  2. layout-only prompt for bboxes -> per-crop vision calls
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .blocks import BLOCK_KINDS, Block, PageModel, block_id_for, page_id_for

log = logging.getLogger(__name__)

NORMALISER = 1000.0

PAGE_PROMPT = """You are a document digitization engine. Analyze this page image.
Return STRICT JSON only, no prose, no markdown fences:
{"caption":"<one-sentence visual description of the page>",
 "regions":[{"bbox":[x0,y0,x1,y1] in 0..1000 normalized coords,
   "kind":"heading|paragraph|math|table|diagram|caption|note",
   "content":"<faithful transcription. math->LaTeX body; table->LaTeX tabular body;
              diagram->TikZ code or precise structural description>",
   "order":<int reading order>}]}
Rules: preserve every symbol exactly; never invent text; blank page -> {"caption":"","regions":[]}.
Kind must be one of: heading, paragraph, math, table, diagram, caption, note."""

JSON_ONLY_SUFFIX = "\nRemember: output ONLY the JSON object. No explanation."

LAYOUT_PROMPT = """Analyze this page image. Return STRICT JSON only:
{"caption":"<one sentence>",
 "regions":[{"bbox":[x0,y0,x1,y1] in 0..1000 normalized coords,
             "kind":"heading|paragraph|math|table|diagram|caption|note",
             "order":<int>}]}
No content transcription, JSON only."""

CROP_PROMPT = """Transcribe ONLY the highlighted region of this page image.
Return STRICT JSON: {"kind":"heading|paragraph|math|table|diagram|caption|note",
"content":"<faithful transcription; math/table -> LaTeX, diagram -> TikZ or description>"}
No prose, no markdown."""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str):
    """Pull the first parseable JSON value out of arbitrary model output."""
    if not text:
        raise ValueError("empty model output")
    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    # whole-text parse first (so a bare array stays an array), then the
    # outermost balanced structure (so prose around the JSON does not break it)
    outer = _balanced_json(text)
    if outer is not None:
        candidates.append(json.dumps(outer))
    last_err: Exception | None = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception as exc:
            last_err = exc
    if outer is not None:
        return outer
    raise ValueError(f"no JSON in model output: {last_err}")


def _balanced_json(text: str):
    """Parse the outermost balanced ``{...}``/``[...]`` in *text*, else None."""
    obj_i, arr_i = text.find("{"), text.find("[")
    brackets = []
    if obj_i >= 0:
        brackets.append((obj_i, "{", "}"))
    if arr_i >= 0:
        brackets.append((arr_i, "[", "]"))
    if not brackets:
        return None
    _, open_ch, close_ch = min(brackets, key=lambda t: t[0])

    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(regions: list["Region"], threshold: float = 0.5) -> list["Region"]:
    """Greedy non-max suppression - duplicated detections collapse."""
    ordered = sorted(regions, key=lambda r: (-r.confidence, r.order,
                                             r.bbox[1], r.bbox[0]))
    kept: list[Region] = []
    for cand in ordered:
        if all(iou(cand.bbox, k.bbox) < threshold for k in kept):
            kept.append(cand)
    # restore a stable visual order for the caller
    kept.sort(key=lambda r: (r.order, r.bbox[1], r.bbox[0]))
    return kept


def reading_order(regions: list["Region"], page_width: float) -> list["Region"]:
    """Column-aware reading order (spec: cluster x-centers, sort (col, y0)).

    Full-width blocks (titles, rules) *start a band* so a two-column page
    reads ``heading -> left column -> right column`` and a page with a
    mid-sheet heading reads ``above -> heading -> below``.
    """
    if not regions:
        return []
    width = page_width or max(r.bbox[2] for r in regions) or 1.0

    def is_full(r: "Region") -> bool:
        return (r.bbox[2] - r.bbox[0]) > 0.6 * width

    bands: list[list[Region]] = []
    current: list[Region] = []
    for region in sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0])):
        if is_full(region):
            if current:
                bands.append(current)
                current = []
            current.append(region)
        else:
            current.append(region)
    if current:
        bands.append(current)

    ordered: list[Region] = []
    for band in bands:
        keep_first = bool(band) and is_full(band[0])
        head = band[:1] if keep_first else []
        rest = band[1:] if keep_first else band
        ordered.extend(head)
        ordered.extend(_sort_columns(rest, width))

    for i, region in enumerate(ordered):
        region.order = i
    return ordered


def _sort_columns(regions: list["Region"], width: float) -> list[Region]:
    if not regions:
        return []
    centers = sorted((r.bbox[0] + r.bbox[2]) / 2.0 for r in regions)
    gap = max(0.12 * width, 1.0)
    breaks = [i for i in range(1, len(centers)) if centers[i] - centers[i - 1] > gap]
    n_cols = len(breaks) + 1
    if n_cols == 1:
        return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))
    # assign each region to a column by its centre
    edges = [centers[0] - 1.0]
    for b in breaks:
        edges.append((centers[b - 1] + centers[b]) / 2.0)
    edges.append(centers[-1] + 1.0)

    def col_of(r: "Region") -> int:
        c = (r.bbox[0] + r.bbox[2]) / 2.0
        for i in range(len(edges) - 1):
            if edges[i] <= c < edges[i + 1]:
                return i
        return len(edges) - 2

    out: list[Region] = []
    for col in range(len(edges) - 1):
        out.extend(sorted((r for r in regions if col_of(r) == col),
                          key=lambda r: (r.bbox[1], r.bbox[0])))
    return out


def rescale(bbox: Iterable[float], width: int, height: int,
            normaliser: float = NORMALISER) -> tuple[float, float, float, float]:
    """0..1000 model coords -> page pixels (clamped inside the page)."""
    vals = [float(v) for v in bbox]
    if len(vals) != 4:
        return (0.0, 0.0, float(width), float(height))
    sx, sy = width / normaliser, height / normaliser
    x0, x1 = sorted((vals[0] * sx, vals[2] * sx))
    y0, y1 = sorted((vals[1] * sy, vals[3] * sy))
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(width), x1), min(float(height), y1)
    if x1 <= x0:
        x1 = min(float(width), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(height), y0 + 1.0)
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# region model
# ---------------------------------------------------------------------------
@dataclass
class Region:
    bbox: tuple[float, float, float, float]
    kind: str
    content: str = ""
    order: int = 0
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    @property
    def frac(self) -> float:      # area relative to the page (set by caller)
        return float(self.meta.get("frac", 0.0))


@dataclass
class ParsedPage:
    caption: str = ""
    regions: list[Region] = field(default_factory=list)
    raw: str = ""
    mode: str = "page"            # page | crops | empty
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.regions) or not self.caption


def _coerce_kind(value, warnings: list[str]) -> str:
    kind = str(value or "").strip().lower()
    if kind in BLOCK_KINDS:
        return kind
    # forgiving mapping for models that paraphrase
    aliases = {"text": "paragraph", "body": "paragraph", "equation": "math",
               "formula": "math", "equations": "math", "title": "heading",
               "subtitle": "heading", "section": "heading", "image": "diagram",
               "figure": "diagram", "drawing": "diagram", "sketch": "diagram",
               "list": "paragraph", "bullet": "paragraph", "note": "note",
               "caption": "caption", "table": "table", "heading": "heading",
               "paragraph": "paragraph", "diagram": "diagram"}
    mapped = aliases.get(kind)
    if mapped:
        return mapped
    warnings.append(f"unknown kind {value!r} -> paragraph")
    return "paragraph"


def parse_page_json(text: str, width: int, height: int, *,
                    nms_iou: float = 0.5, min_box_frac: float = 0.01,
                    ) -> ParsedPage:
    """Model output -> :class:`ParsedPage` with pixel-space regions."""
    warnings: list[str] = []
    data = extract_json(text)
    if isinstance(data, list):                     # bare array of regions
        data = {"caption": "", "regions": data}
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")

    caption = str(data.get("caption") or "").strip()
    raw_regions = data.get("regions") or data.get("blocks") or []
    if not isinstance(raw_regions, list):
        raise ValueError("regions must be a list")

    regions: list[Region] = []
    page_area = float(max(1, width) * max(1, height))
    for i, item in enumerate(raw_regions):
        if not isinstance(item, dict):
            warnings.append(f"region {i} is not an object")
            continue
        raw_bbox = item.get("bbox") or item.get("box") or [0, 0, 0, 0]
        if isinstance(raw_bbox, dict):
            raw_bbox = [raw_bbox.get(k, 0) for k in ("x0", "y0", "x1", "y1")]
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            warnings.append(f"region {i} has bad bbox {raw_bbox!r}")
            continue
        bbox = rescale(raw_bbox, width, height)
        region = Region(
            bbox=bbox, kind=_coerce_kind(item.get("kind"), warnings),
            content=str(item.get("content") or item.get("text") or ""),
            order=int(item.get("order", i) if item.get("order") is not None else i),
            confidence=float(item.get("confidence", 1.0) or 1.0),
            meta={"frac": region_frac(bbox, page_area)},
        )
        if region.frac < min_box_frac:
            warnings.append(f"dropped tiny box ({region.frac:.3%})")
            continue
        regions.append(region)

    regions = nms(regions, threshold=nms_iou)
    regions = reading_order(regions, float(width))

    if not regions and not caption and raw_regions:
        raise ValueError("all regions were rejected")
    return ParsedPage(caption=caption, regions=regions, raw=text,
                      mode="page", warnings=warnings)


def region_frac(bbox: tuple[float, ...], page_area: float) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / page_area


# ---------------------------------------------------------------------------
# fallback: layout prompt + per-crop calls
# ---------------------------------------------------------------------------
CropFn = Callable[[object, str], str]     # (image, prompt) -> raw model text


def parse_with_crops(image, layout_text: str, width: int, height: int,
                     crop_fn: CropFn, *, nms_iou: float = 0.5,
                     min_box_frac: float = 0.01) -> ParsedPage:
    """Degraded path: bboxes from a layout-only prompt, text from crop calls.

    Guarantees a usable result when the model refuses to emit page-level JSON
    (the single most common VLM failure mode).
    """
    warnings: list[str] = []
    layout = extract_json(layout_text)
    if isinstance(layout, dict):
        caption = str(layout.get("caption") or "").strip()
        raw_regions = layout.get("regions") or []
    else:
        caption, raw_regions = "", layout

    regions: list[Region] = []
    page_area = float(max(1, width) * max(1, height))
    for i, item in enumerate(raw_regions if isinstance(raw_regions, list) else []):
        raw_bbox = item.get("bbox") if isinstance(item, dict) else None
        if not raw_bbox or len(raw_bbox) != 4:
            continue
        bbox = rescale(raw_bbox, width, height)
        region = Region(bbox=bbox,
                        kind=_coerce_kind(item.get("kind") if isinstance(item, dict)
                                          else "paragraph", warnings),
                        order=i, meta={"frac": region_frac(bbox, page_area)})
        if region.frac < min_box_frac:
            continue
        regions.append(region)
    regions = nms(regions, threshold=nms_iou)
    regions = reading_order(regions, float(width))

    for region in regions:
        try:
            from .preprocess import crop_region
            crop = crop_region(image, region.bbox, pad=0.03)
            out = crop_fn(crop, CROP_PROMPT)
            data = extract_json(out)
            region.content = str(data.get("content") or "")
            if data.get("kind"):
                region.kind = _coerce_kind(data["kind"], warnings)
        except Exception as exc:                    # keep going: partial > nothing
            warnings.append(f"crop failed {region.bbox}: {exc}")
    mode = "crops" if any(r.content for r in regions) else "empty"
    return ParsedPage(caption=caption, regions=regions, raw=layout_text,
                      mode=mode, warnings=warnings)


# ---------------------------------------------------------------------------
# regions -> blocks
# ---------------------------------------------------------------------------
def regions_to_blocks(parsed: ParsedPage, doc_id: str, page_no: int,
                      page_id: str | None = None,
                      seq_offset: int = 0) -> list[Block]:
    page_id = page_id or page_id_for(doc_id, page_no)
    blocks: list[Block] = []
    for i, region in enumerate(parsed.regions):
        seq = seq_offset + i
        is_text = region.kind in ("paragraph", "caption", "note", "heading")
        content = region.content.strip()
        blocks.append(Block(
            block_id=block_id_for(doc_id, seq), page_id=page_id, doc_id=doc_id,
            seq=seq, kind=region.kind, bbox=region.bbox,
            text=content if is_text else "",
            latex="" if is_text else content,
            confidence=region.confidence,
            meta={"source": parsed.mode, "page_no": page_no},
            page_no=page_no))
    return blocks


def fill_page(page: PageModel, parsed: ParsedPage, doc_id: str,
              seq_offset: int = 0) -> list[Block]:
    blocks = regions_to_blocks(parsed, doc_id, page.page_no, page.page_id,
                               seq_offset)
    page.blocks.extend(blocks)
    page.caption = parsed.caption or page.caption
    return blocks
