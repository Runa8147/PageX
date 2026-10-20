"""WP9 - page viewer: raster + block overlay, citation highlight.

The overlay is drawn from the ``blocks`` table (bbox is 0..1000 normalized),
which is what the rest of the pipeline treats as ground truth for geometry.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPixmap, QPen
from PyQt6.QtWidgets import (QGraphicsRectItem, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

log = logging.getLogger(__name__)

KIND_COLORS = {
    "heading": "#f59e0b",
    "paragraph": "#3b82f6",
    "math": "#a855f7",
    "table": "#14b8a6",
    "diagram": "#ef4444",
    "caption": "#64748b",
    "note": "#22c55e",
}


class PageViewer(QWidget):
    """Center pane: the page image with selectable block rectangles."""

    blockSelected = pyqtSignal(str)               # block_id
    pageChanged = pyqtSignal(int)                 # page_no (1-based)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: list[dict] = []
        self._blocks: list[dict] = []
        self._index = 0
        self._rects: dict[str, QGraphicsRectItem] = {}
        self._selected: str | None = None
        self._build()

    # -- construction ----------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(44)
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(44)
        self.prev_btn.clicked.connect(lambda: self.goto(self._index - 1))
        self.next_btn.clicked.connect(lambda: self.goto(self._index + 1))
        self.page_label = QLabel("no document selected")
        self.page_label.setObjectName("pageTitle")
        self.caption_label = QLabel("")
        self.caption_label.setObjectName("muted")
        self.zoom_label = QLabel("")
        self.zoom_label.setObjectName("muted")
        bar.addWidget(self.prev_btn)
        bar.addWidget(self.next_btn)
        bar.addWidget(self.page_label)
        bar.addWidget(self.caption_label, 1)
        bar.addWidget(self.zoom_label)
        layout.addLayout(bar)

        self.scene = QGraphicsScene(self)
        self.view = _ZoomView(self.scene)
        self.view.clicked.connect(self._on_scene_click)
        from PyQt6.QtGui import QPainter
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.view, 1)

        legend = QHBoxLayout()
        legend.setSpacing(10)
        for kind, color in KIND_COLORS.items():
            sw = QLabel(f'<span style="color:{color}">■</span> {kind}')
            sw.setObjectName("muted")
            legend.addWidget(sw)
        legend.addStretch(1)
        self.sel_label = QLabel("")
        self.sel_label.setObjectName("muted")
        legend.addWidget(self.sel_label)
        layout.addLayout(legend)
        self.refresh_theme()

    # -- data ------------------------------------------------------------
    def set_document(self, pages: list[dict], blocks: list[dict]) -> None:
        """*pages* from ``Store.get_pages``, *blocks* from ``get_blocks``."""
        self._pages = list(pages or [])
        self._blocks = list(blocks or [])
        self._index = 0
        self.prev_btn.setEnabled(len(self._pages) > 1)
        self.next_btn.setEnabled(len(self._pages) > 1)
        if not self._pages:
            self.page_label.setText("no pages")
            self.caption_label.setText("")
            self.scene.clear()
            self._rects = {}
            return
        self.goto(0)

    def goto(self, index: int) -> None:
        if not self._pages:
            return
        index = max(0, min(index, len(self._pages) - 1))
        self._index = index
        page = self._pages[index]
        self._render(page)
        self.page_label.setText(
            f"page {page['page_no']} / {self._pages[-1]['page_no']}")
        meta = page.get("meta") or {}
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta or "{}")
            except Exception:
                meta = {}
        self.caption_label.setText(str(meta.get("caption", ""))[:120])
        self.pageChanged.emit(int(page["page_no"]))

    def current_page(self) -> dict | None:
        return self._pages[self._index] if self._pages else None

    # -- rendering -------------------------------------------------------
    def _render(self, page: dict) -> None:
        self.scene.clear()
        self._rects = {}
        path = page.get("image_path")
        pix = QPixmap()
        if path:
            resolved = Path(str(path))
            if not resolved.is_absolute():
                from ..core.config import CFG
                resolved = CFG.artifacts_dir / resolved
            if resolved.exists():
                pix.load(str(resolved))
        if pix.isNull():
            w = int(page.get("width") or 800)
            h = int(page.get("height") or 1131)
            pix = QPixmap(w, h)
            pix.fill(QColor("#fafafa"))
        self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))
        w, h = pix.width(), pix.height()

        for blk in self._blocks:
            if blk.get("page_id") != page.get("page_id"):
                continue
            x0, y0, x1, y1 = [float(v) for v in blk["bbox"]]
            rect = QRectF(x0 / 1000.0 * w, y0 / 1000.0 * h,
                          max(2.0, (x1 - x0) / 1000.0 * w),
                          max(2.0, (y1 - y0) / 1000.0 * h))
            color = QColor(KIND_COLORS.get(blk.get("kind", ""), "#6b7280"))
            item = self.scene.addRect(rect, QPen(color, 1.4),
                                      QBrush(QColor(color.red(), color.green(),
                                                   color.blue(), 28)))
            item.setZValue(10)
            item.setData(0, blk.get("block_id", ""))
            tip = f'{blk.get("kind", "")} · {blk.get("block_id", "")}\n{str(blk.get("text", ""))[:200]}'
            item.setToolTip(tip)
            self._rects[blk.get("block_id", "")] = item
        self.view.fit()

    # -- interaction -----------------------------------------------------
    def _on_scene_click(self, x: float, y: float) -> None:
        # topmost rect under the point (rects sit at ZValue 10)
        for it in self.scene.items(QPointF(x, y)):
            if isinstance(it, QGraphicsRectItem) and it.data(0):
                self.highlight_block(str(it.data(0)), select=True)
                return
        self.sel_label.setText("")

    def highlight_block(self, block_id: str, *, select: bool = False) -> None:
        """Flash/select a block rectangle (used by editor + citations)."""
        for bid, item in self._rects.items():
            sel = (bid == block_id)
            pen = item.pen()
            pen.setWidthF(2.6 if sel and select else 1.4)
            item.setPen(pen)
        item = self._rects.get(block_id)
        if item is None:
            return
        self.view.ensureVisible(item)
        if select and self._selected != block_id:
            self._selected = block_id
            self.sel_label.setText(f"selected: {block_id}")
            self.blockSelected.emit(block_id)

    def highlight_blocks(self, block_ids: list[str]) -> None:
        """Draw citation highlights (multi-block) without firing selection."""
        for bid, item in self._rects.items():
            active = bid in block_ids
            pen = item.pen()
            pen.setWidthF(2.8 if active else 1.4)
            pen.setColor(QColor("#dc2626") if active else
                         QColor(KIND_COLORS.get(_kind_of(self._blocks, bid), "#6b7280")))
            item.setPen(pen)
            if active:
                self.view.ensureVisible(item)
        if block_ids:
            self.sel_label.setText(f"highlighted {len(block_ids)} block(s)")

    def show_citation(self, citation: dict) -> None:
        """Jump to the cited page and highlight its blocks."""
        page_id = citation.get("page_id") or ""
        for i, page in enumerate(self._pages):
            if page["page_id"] == page_id:
                self.goto(i)
                break
        self.highlight_blocks([str(b) for b in (citation.get("block_ids") or [])])

    def refresh_theme(self) -> None:
        """Re-apply canvas background for the active light/dark theme."""
        try:
            from PyQt6.QtWidgets import QApplication
            from .theme import is_dark
            dark = is_dark(QApplication.instance())
        except Exception:
            dark = False
        self.view.setBackgroundBrush(
            QColor("#0b1220") if dark else QColor("#e8edf3"))


def _kind_of(blocks: list[dict], block_id: str) -> str:
    for b in blocks:
        if b.get("block_id") == block_id:
            return str(b.get("kind", ""))
    return ""


class _ZoomView(QGraphicsView):
    """GraphicsView that reports plain clicks and supports wheel-zoom."""

    clicked = pyqtSignal(float, float)              # scene coordinates

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._zoom = 1.0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        try:
            from PyQt6.QtWidgets import QApplication
            from .theme import is_dark
            dark = is_dark(QApplication.instance())
        except Exception:
            dark = False
        self.setBackgroundBrush(QColor("#0b1220" if dark else "#e8edf3"))

    def wheelEvent(self, event):                    # noqa: N802 - Qt API
        delta = event.angleDelta().y()
        factor = 1.2 if delta > 0 else 1 / 1.2
        new_zoom = max(0.2, min(6.0, self._zoom * factor))
        self.scale(new_zoom / self._zoom, new_zoom / self._zoom)
        self._zoom = new_zoom

    def fit(self) -> None:
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0

    def resizeEvent(self, event):                   # noqa: N802 - Qt API
        super().resizeEvent(event)
        # The viewer is often populated before the window is shown; keep the
        # page fitted to the canvas until the user explicitly zooms.
        if self._zoom == 1.0 and not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):               # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            sp = self.mapToScene(event.position().toPoint())
            self.clicked.emit(sp.x(), sp.y())
        super().mousePressEvent(event)
