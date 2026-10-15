"""Synthetic fixtures: sample pages and canned model outputs.

The evaluation dataset is a *deliverable* (per the build manual), so the
samples are generated deterministically here rather than committed as binary
blobs - ``python -m pagex.testing.fixtures`` regenerates ``samples/``.

``HEAVY_FONT`` hunt: a handwriting-style face if one exists on the machine,
otherwise DejaVu Sans rendered with per-glyph jitter/rotation so strokes do not
look typeset.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core import paths

ROOT = paths.repo_root() or paths.base_dir()
SAMPLES = ROOT / "samples"

HAND_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    r"C:\Windows\Fonts\comic.ttf",            # Windows: handwriting-ish
    r"C:\Windows\Fonts\segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
]
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
FONT_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
MONO_FONT = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    r"C:\Windows\Fonts\consola.ttf",          # Windows: Consolas
    "/System/Library/Fonts/Menlo.ttc",
]

A4 = (1240, 1754)          # 150 dpi


def _font(paths: list[str], size: int) -> ImageFont.FreeTypeFont:
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def _jitter_text(draw: ImageDraw.ImageDraw, xy, text, font, fill,
                 rng: random.Random, jitter: float = 1.6, rot: float = 4.0):
    """Draw *text* with per-glyph offset/rotation noise (handwriting-ish)."""
    x, y = xy
    for ch in text:
        if ch == " ":
            x += font.getlength(" ")
            continue
        dx = rng.uniform(-jitter, jitter)
        dy = rng.uniform(-jitter, jitter)
        ang = rng.uniform(-rot, rot)
        glyph = Image.new("RGBA", (int(font.getlength(ch)) + 8, font.size + 12),
                          (255, 255, 255, 0))
        gdraw = ImageDraw.Draw(glyph)
        gdraw.text((4, 4), ch, font=font, fill=fill)
        if abs(ang) > 0.05:
            glyph = glyph.rotate(ang, resample=Image.BICUBIC, expand=True)
        draw._image.paste(glyph, (int(x + dx), int(y + dy - 6)), glyph)
        x += font.getlength(ch) + rng.uniform(0.2, 1.4)


# ---------------------------------------------------------------------------
# page generators
# ---------------------------------------------------------------------------
def handwritten_page(lines: list[tuple[str, str]] | None = None,
                     size: tuple[int, int] = A4, seed: int = 7,
                     title: str = "Physics - Rotational Motion",
                     ) -> Image.Image:
    """A4 page of "handwritten" notes: heading + paragraphs + equations."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, (252, 251, 247))
    draw = ImageDraw.Draw(img)
    # faint ruled lines, like a notebook
    for y in range(120, size[1] - 40, 46):
        draw.line([(70, y), (size[0] - 70, y)], fill=(226, 232, 240), width=1)
    draw.line([(90, 60), (90, size[1] - 50)], fill=(244, 200, 200), width=2)

    head = _font(FONT_BOLD, 46)
    body = _font(HAND_FONT_CANDIDATES, 34)
    mathf = _font(MONO_FONT, 36)

    if lines is None:
        lines = [
            ("heading", title),
            ("text", "Rotational dynamics describes how objects spin when a"),
            ("text", "torque is applied about a fixed axis of rotation."),
            ("math", "tau = I * alpha"),
            ("text", "The moment of inertia I depends on how mass is distributed:"),
            ("math", "I = sum(m_i * r_i^2)"),
            ("text", "For a solid cylinder of mass M and radius R we get:"),
            ("math", "I = (1/2) * M * R^2"),
            ("text", "Kinetic energy of a rotating body is therefore:"),
            ("math", "K = (1/2) * I * omega^2"),
            ("text", "Angular momentum is conserved when no external torque acts."),
            ("math", "L = I * omega = constant"),
            ("text", "Worked example: a figure skater pulls arms in and spins faster"),
            ("text", "because I decreases while L stays the same."),
            ("note", "Remember: torque and angular acceleration are vectors!"),
        ]

    y = 70
    for kind, text in lines:
        if kind == "heading":
            _jitter_text(draw, (110, y), text, head, (26, 32, 44), rng,
                         jitter=1.0, rot=1.5)
            y += 70
        elif kind == "math":
            draw.rectangle([105, y - 6, size[0] - 110, y + 54],
                           fill=(238, 240, 252))
            _jitter_text(draw, (130, y), text, mathf, (40, 44, 90), rng,
                         jitter=0.4, rot=0.6)
            y += 68
        elif kind == "note":
            _jitter_text(draw, (120, y), text, body, (150, 60, 40), rng)
            y += 56
        else:
            _jitter_text(draw, (110, y), text, body, (34, 40, 52), rng)
            y += 52
        if y > size[1] - 90:
            break
    return img


def formula_page(size: tuple[int, int] = A4, seed: int = 11) -> Image.Image:
    """Equation-heavy page - the fixture TC-INT-01 expects `kind=math`."""
    return handwritten_page(seed=seed, title="Mathematics - Calculus",
                            lines=[
                                ("heading", "Calculus - Derivatives"),
                                ("text", "The derivative of a function measures its rate of change."),
                                ("math", "d/dx (x^n) = n * x^(n-1)"),
                                ("text", "Product rule for two functions u and v:"),
                                ("math", "(u*v)' = u'*v + u*v'"),
                                ("text", "Chain rule:"),
                                ("math", "d/dx f(g(x)) = f'(g(x)) * g'(x)"),
                                ("text", "Definite integral gives the area under the curve:"),
                                ("math", "int_a^b f(x) dx = F(b) - F(a)"),
                                ("text", "Euler's identity links five constants:"),
                                ("math", "e^(i*pi) + 1 = 0"),
                            ])


def two_column_page(size: tuple[int, int] = A4, seed: int = 3) -> Image.Image:
    """Full-width heading over two columns (reading-order test fixture)."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    head = _font(FONT_BOLD, 44)
    body = _font(FONT_CANDIDATES, 26)
    _jitter_text(draw, (70, 60), "Two Column Layout Sample", head, (0, 0, 0), rng,
                 jitter=0.3, rot=0.5)
    col_w = (size[0] - 200) // 2
    for col, x0 in enumerate((70, 110 + col_w)):
        y = 170
        for i in range(12):
            text = f"Column {col + 1} paragraph {i + 1} of layout text."
            _jitter_text(draw, (x0, y), text, body, (20, 20, 20), rng)
            y += 44
    return img


def table_page(size: tuple[int, int] = A4, seed: int = 5) -> Image.Image:
    """Typed table + caption - exercises table -> \\begin{tabular}."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    head = _font(FONT_BOLD, 40)
    body = _font(FONT_CANDIDATES, 28)
    draw.text((70, 60), "Experiment Results", font=head, fill=(0, 0, 0))
    rows = [["Trial", "Mass (kg)", "Time (s)", "Energy (J)"],
            ["1", "0.50", "1.20", "0.31"],
            ["2", "0.75", "1.05", "0.49"],
            ["3", "1.00", "0.92", "0.71"],
            ["4", "1.25", "0.85", "0.98"]]
    x0, y0, w, h = 70, 150, 1100, 64
    for r, row in enumerate(rows):
        y = y0 + r * h
        for c, cell in enumerate(row):
            x = x0 + c * (w // len(row))
            draw.rectangle([x, y, x + w // len(row), y + h],
                           outline=(60, 60, 60), width=2,
                           fill=(235, 238, 245) if r == 0 else None)
            draw.text((x + 14, y + 16), cell, font=body, fill=(0, 0, 0))
    draw.text((70, y0 + len(rows) * h + 30),
              "Table 1: measured energy rises with mass.",
              font=body, fill=(70, 70, 70))
    return img


def form_page(size: tuple[int, int] = A4) -> Image.Image:
    """Registration form: labels + underlines (layout segmentation demo)."""
    img = Image.new("RGB", size, (255, 255, 253))
    draw = ImageDraw.Draw(img)
    head = _font(FONT_BOLD, 42)
    body = _font(FONT_CANDIDATES, 30)
    draw.text((70, 60), "Student Registration Form", font=head, fill=(0, 0, 0))
    fields = ["Name:", "Roll Number:", "Department:", "Email:", "Semester:"]
    y = 170
    for f in fields:
        draw.text((80, y), f, font=body, fill=(0, 0, 0))
        draw.line([(330, y + 34), (size[0] - 90, y + 34)], fill=(30, 30, 30),
                  width=2)
        y += 96
    draw.text((80, y + 20), "Signature of the applicant:", font=body, fill=(0, 0, 0))
    draw.line([(80, y + 100), (600, y + 100)], fill=(30, 30, 30), width=2)
    return img


def diagram_page(size: tuple[int, int] = A4) -> Image.Image:
    """Hand-drawn block diagram ( -> TikZ / structural description)."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    head = _font(FONT_BOLD, 38)
    body = _font(FONT_CANDIDATES, 26)
    draw.text((70, 60), "System Architecture", font=head, fill=(0, 0, 0))
    boxes = [("Camera", 120), ("Preprocess", 420), ("VLM", 720), ("LaTeX", 1020)]
    for i, (label, x) in enumerate(boxes):
        y = 300
        draw.rectangle([x, y, x + 210, y + 110], outline=(20, 20, 20), width=3)
        draw.text((x + 30, y + 40), label, font=body, fill=(0, 0, 0))
        if i < len(boxes) - 1:
            draw.line([(x + 210, y + 55), (x + 300, y + 55)], fill=(0, 0, 0),
                      width=3)
            draw.polygon([(x + 300, y + 55), (x + 286, y + 47),
                          (x + 286, y + 63)], fill=(0, 0, 0))
    draw.text((70, 520), "Arrow: data flows left to right, each stage is a module.",
              font=body, fill=(40, 40, 40))
    # a small triangle under the boxes (extra shape for the segmenter)
    draw.polygon([(420, 640), (620, 640), (520, 780)], outline=(0, 0, 0))
    return img


def rotated_page(degrees: float = 7.0, size: tuple[int, int] = (900, 1200),
                 seed: int = 21) -> Image.Image:
    """Deskew fixture: a ruled page rotated by *degrees*."""
    img = handwritten_page(size=size, seed=seed, title="Deskew Test",
                           lines=[("heading", "Deskew Test Page"),
                                  ("text", "This line should end up horizontal."),
                                  ("text", "Second line of the calibration text."),
                                  ("math", "y = m * x + b"),
                                  ("text", "Third line used to estimate skew.")])
    return img.rotate(degrees, resample=Image.BICUBIC, expand=True,
                      fillcolor=(252, 251, 247))


def small_note(size: tuple[int, int] = (700, 990), seed: int = 13) -> Image.Image:
    """Sub-1500 px note: exercises the 2x Lanczos upscale path."""
    return handwritten_page(size=size, seed=seed, title="Short Note",
                            lines=[("heading", "Short Note"),
                                   ("text", "Small image that must be upscaled."),
                                   ("math", "E = m * c^2"),
                                   ("text", "CLAHE should lift the pencil contrast.")])


# ---------------------------------------------------------------------------
# canned model output (unit tests + smoke + fallback demo)
# ---------------------------------------------------------------------------
CANNED_PAGE_JSON = json.dumps({
    "caption": "Handwritten physics notes with three equations and a closing remark.",
    "regions": [
        {"bbox": [70, 40, 930, 110], "kind": "heading", "order": 0,
         "content": "Physics - Rotational Motion"},
        {"bbox": [70, 130, 930, 250], "kind": "paragraph", "order": 1,
         "content": "Rotational dynamics describes how objects spin when a torque is "
                    "applied about a fixed axis of rotation."},
        {"bbox": [80, 270, 560, 350], "kind": "math", "order": 2,
         "content": "\\tau = I \\alpha"},
        {"bbox": [70, 380, 930, 500], "kind": "paragraph", "order": 3,
         "content": "The moment of inertia I depends on how mass is distributed "
                    "from the axis."},
        {"bbox": [90, 520, 700, 600], "kind": "math", "order": 4,
         "content": "I = \\sum_{i} m_i r_i^2"},
        {"bbox": [70, 640, 930, 760], "kind": "paragraph", "order": 5,
         "content": "Angular momentum is conserved when no external torque acts."},
        {"bbox": [90, 790, 640, 870], "kind": "math", "order": 6,
         "content": "L = I \\omega"},
        {"bbox": [70, 900, 930, 980], "kind": "note", "order": 7,
         "content": "Remember: torque and angular acceleration are vectors!"},
    ],
}, indent=2)

CANNED_LAYOUT_JSON = json.dumps({
    "caption": "Handwritten physics notes with equations.",
    "regions": [
        {"bbox": [70, 40, 930, 110], "kind": "heading", "order": 0},
        {"bbox": [70, 130, 930, 250], "kind": "paragraph", "order": 1},
        {"bbox": [80, 270, 560, 350], "kind": "math", "order": 2},
    ],
})

CANNED_CROP_JSON = json.dumps({
    "kind": "math",
    "content": "\\tau = I \\alpha",
})

# table-shaped canned output for exporter tests
CANNED_TABLE_JSON = json.dumps({
    "caption": "Typed results table with a caption.",
    "regions": [
        {"bbox": [70, 40, 930, 110], "kind": "heading", "order": 0,
         "content": "Experiment Results"},
        {"bbox": [70, 160, 940, 520], "kind": "table", "order": 1,
         "content": "\\begin{tabular}{llll}\nTrial & Mass (kg) & Time (s) & Energy (J) \\\\\n"
                    "1 & 0.50 & 1.20 & 0.31 \\\\\n2 & 0.75 & 1.05 & 0.49 \\\\\n"
                    "\\end{tabular}"},
        {"bbox": [70, 560, 940, 620], "kind": "caption", "order": 2,
         "content": "Table 1: measured energy rises with mass."},
    ],
})


def write_samples(target: Path | None = None) -> dict[str, Path]:
    """Materialise the sample corpus used by eval + the demo runbook.

    Default location follows the platform ladder (``PAGEX_HOME`` -> bundled
    -> repository checkout -> user data dir); pass *target* to override.
    """
    target = Path(target) if target else paths.samples_dir(create=True)
    (target / "notes").mkdir(parents=True, exist_ok=True)
    (target / "digital").mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def save(img: Image.Image, rel: str) -> Path:
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        written[rel] = p
        return p

    save(handwritten_page(seed=7), "notes/1_rotational_motion.png")
    save(handwritten_page(seed=21, title="Thermodynamics - Entropy",
                          lines=None), "notes/2_entropy.png")
    save(formula_page(), "notes/3_calculus.png")
    save(table_page(), "digital/table.png")
    save(form_page(), "digital/form.png")
    save(diagram_page(), "digital/diagram.png")
    save(two_column_page(), "digital/two_column.png")
    save(small_note(), "notes/small_note.png")

    # one born-digital PDF (self-made, so we never redistribute copyrighted text)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
        pdf_path = target / "digital/page_sample.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 20)
        c.drawString(2 * cm, h - 2 * cm, "PageX Sample Document")
        c.setFont("Helvetica", 12)
        body = [
            "This born-digital PDF exercises the PyMuPDF render path.",
            "Hybrid search combines BM25 keywords with cosine similarity.",
            "The knowledge graph links concepts across every ingested page.",
            "Latent semantic structure survives LaTeX round-tripping.",
            "Retrieval is evaluated with Recall@5 and MRR on queries.json.",
            "",
            "E = m c^2      integral_a^b f(x) dx = F(b) - F(a)",
        ]
        y = h - 3 * cm
        for line in body:
            c.drawString(2 * cm, y, line)
            y -= 0.75 * cm
        c.setFont("Helvetica-Oblique", 11)
        c.drawString(2 * cm, y - 0.5 * cm,
                     "Generated by pagex.testing.fixtures - no third-party content.")
        c.showPage()
        c.save()
        written["digital/page_sample.pdf"] = pdf_path
    except Exception as exc:                                # pragma: no cover
        print(f"pdf sample skipped: {exc}")

    queries = [
        {"query": "What is the moment of inertia of a solid cylinder?",
         "expected_doc": "notes/3_calculus.png", "expected_page": 1,
         "tags": ["math", "formula"]},
        {"query": "How does torque relate to angular acceleration?",
         "expected_doc": "notes/1_rotational_motion.png", "expected_page": 1,
         "tags": ["physics"]},
        {"query": "state the chain rule for derivatives",
         "expected_doc": "notes/3_calculus.png", "expected_page": 1},
        {"query": "angular momentum conservation",
         "expected_doc": "notes/1_rotational_motion.png", "expected_page": 1},
        {"query": "measured energy table with mass and time",
         "expected_doc": "digital/table.png", "expected_page": 1},
        {"query": "student registration form fields",
         "expected_doc": "digital/form.png", "expected_page": 1},
        {"query": "system architecture camera to latex pipeline",
         "expected_doc": "digital/diagram.png", "expected_page": 1},
        {"query": "hybrid search bm25 cosine similarity",
         "expected_doc": "digital/page_sample.pdf", "expected_page": 1},
        {"query": "recall@5 and mrr evaluation metric",
         "expected_doc": "digital/page_sample.pdf", "expected_page": 1},
        {"query": "entropy and thermodynamics second law",
         "expected_doc": "notes/2_entropy.png", "expected_page": 1},
        {"query": "euler identity e i pi",
         "expected_doc": "notes/3_calculus.png", "expected_page": 1},
        {"query": "kinetic energy rotating body omega",
         "expected_doc": "notes/1_rotational_motion.png", "expected_page": 1},
    ]
    qpath = target / "queries.json"
    qpath.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    written["queries.json"] = qpath
    return written


def main() -> int:
    written = write_samples()
    for rel, path in written.items():
        print(f"{rel:34} {path.stat().st_size / 1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
