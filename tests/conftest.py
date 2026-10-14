"""Shared pytest fixtures: temporary DB, canned VLM/LLM, synthetic pages."""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagex.core.store import Store            # noqa: E402
from pagex.testing import fixtures            # noqa: E402


class FakeResult:
    """Duck-typed :class:`pagex.core.models.GenResult` for canned model I/O."""

    def __init__(self, text: str):
        self.text = text
        self.tokens = 0
        self.seconds = 0.0
        self.tok_per_s = 0.0

    def __bool__(self) -> bool:
        return bool(self.text)


@pytest.fixture()
def tmp_store(tmp_path) -> Store:
    store = Store(tmp_path / "test.db")
    yield store
    store.close_all()


@pytest.fixture()
def canned_vision():
    """vision_fn that serves canned VLM JSON (no GGUF, no RAM)."""
    def _vision(image, prompt, payload: str | None = None):
        if payload is not None:
            return FakeResult(payload)
        if "layout" in prompt.lower() or "regions" in prompt and "content" not in prompt:
            return FakeResult(fixtures.CANNED_LAYOUT_JSON)
        if "highlighted region" in prompt or "ONLY the highlighted" in prompt:
            return FakeResult(fixtures.CANNED_CROP_JSON)
        return FakeResult(fixtures.CANNED_PAGE_JSON)
    return _vision


@pytest.fixture()
def canned_generate():
    """generate_fn(...) -> canned text (KG extraction, RAG)."""
    def _generate(*args, **kwargs):
        payload = kwargs.pop("payload", None)
        if payload is None:
            prompt = args[0] if args and isinstance(args[0], str) else ""
            if "key concepts" in prompt:
                payload = json.dumps({
                    "concepts": ["moment of inertia", "angular momentum",
                                 "torque"],
                    "edges": [["moment of inertia", "CO_OCCURS", "torque"],
                              ["torque", "PRECEDES", "angular momentum"]]})
            else:
                payload = ("The moment of inertia depends on mass distribution. "
                           "[doc:Physics p.1]")
        return FakeResult(payload)
    return _generate


@pytest.fixture()
def canned_embed():
    """Deterministic 384-dim embedder (no GGUF): bag-of-chars hashing."""
    def _embed(texts):
        out = []
        for t in texts:
            v = np.zeros(384, dtype=np.float32)
            words = [w for w in (t or "").lower().split() if w]
            if not words:
                words = ["empty"]
            for w in words:
                idx = zlib.crc32(w.encode("utf-8")) % 384
                v[idx] += 1.0
            v /= (np.linalg.norm(v) or 1.0)
            out.append(v.tolist())
        return out
    return _embed


@pytest.fixture()
def sample_page():
    """RGB A4-ish synthetic handwritten page (numpy array)."""
    img = fixtures.handwritten_page(seed=7)
    return np.array(img)


@pytest.fixture()
def rotated_sample():
    return np.array(fixtures.rotated_page(7.0))


def resolve_sample_doc(store: Store, fragment: str) -> str | None:
    rows = store.conn.execute("SELECT doc_id, source_path FROM documents").fetchall()
    for r in rows:
        if fragment in (r["source_path"] or ""):
            return r["doc_id"]
    return None
