"""WP3 - the model layer: one heavy slot, one resident embedder.

ADR-002 / ADR-008
------------------
* All models are GGUF and loaded through ``llama-cpp-python`` (explicit
  ``n_threads``, quantised KV cache, real ``del``-based unload).
* The **VLM and the LLM are mutually exclusive**: :class:`ModelSlot` holds at
  most one heavy model at a time.  Ingest never chats, chat never ingests.
  This single rule is why PageX fits 8 GB where ``transformers`` + fp16 OOMed.
* The 384-dim MiniLM embedder is tiny and therefore stays resident.

Nothing here is imported at package import time: unit tests monkeypatch
:func:`generate` / :func:`vision_generate` and never touch a GGUF.
"""
from __future__ import annotations

import base64
import gc
import logging
import mimetypes
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import psutil

from .config import CFG

log = logging.getLogger(__name__)

_proc = psutil.Process()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rss_bytes() -> int:
    try:
        return _proc.memory_info().rss
    except Exception:                                  # pragma: no cover
        return 0


def rss_gb() -> float:
    return rss_bytes() / 1e9


def _data_uri(p: Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(p).read_bytes()).decode()}"


def _image_uri(image: "Path | np.ndarray", max_side: int | None = None) -> str:
    """Path or ndarray -> PNG data URI, long edge clamped to *max_side*.

    Handing the VLM a 3000 px phone photo is the classic way to blow the KV
    cache; clamping here means callers cannot forget it.
    """
    max_side = max_side or CFG.runtime_vlm_max_side
    if isinstance(image, (str, Path)):
        from PIL import Image
        with Image.open(image) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.LANCZOS)
            from io import BytesIO
            buf = BytesIO()
            im.save(buf, format="PNG")
            raw = buf.getvalue()
    else:
        import cv2
        img = image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img.shape[:2]
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not ok:                                     # pragma: no cover
            raise RuntimeError("PNG encode failed")
        raw = buf.tobytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


@dataclass
class GenResult:
    text: str
    tokens: int = 0
    seconds: float = 0.0
    tok_per_s: float = 0.0
    peak_rss_gb: float = 0.0
    meta: dict = field(default_factory=dict)

    def __bool__(self) -> bool:                        # `if result:` in callers
        return bool(self.text)


# ---------------------------------------------------------------------------
# the heavy slot (ADR-002)
# ---------------------------------------------------------------------------
class ModelSlot:
    """Exactly one resident heavy model (VLM *or* LLM), never both.

    ``acquire()`` silently evicts the previously loaded model, which is what
    keeps peak RSS inside the 6.2 GB budget on an 8 GB machine.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._m = None
        self._key: str | None = None
        self._path: Path | None = None
        self._loaded_at = 0.0
        self._load_seconds = 0.0
        self._loaded_rss = 0.0
        self.last_gen: GenResult | None = None
        self.loads = 0
        self.evictions = 0

    # -- introspection ---------------------------------------------------
    @property
    def key(self) -> str | None:
        return self._key

    @property
    def loaded(self) -> bool:
        return self._m is not None

    @property
    def resident_gb(self) -> float:
        return max(0.0, (rss_bytes() - self._loaded_rss) / 1e9) if self._m else 0.0

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._loaded_at if self._m else 0.0

    def state(self) -> str:
        """Status-bar string - live proof of ADR-002 during the demo."""
        if self._m is None:
            return "no model loaded"
        gen = self.last_gen
        speed = f" · {gen.tok_per_s:.1f} tok/s" if gen and gen.tok_per_s else ""
        return (f"{self.key} loaded · {self.resident_gb:.1f} GB resident"
                f" · load {self._load_seconds:.1f}s{speed}")

    # -- lifecycle -------------------------------------------------------
    def acquire(self, key: str, path: Path | str, mmproj: Path | str | None = None,
                n_ctx: int = 4096, *, extra: dict | None = None):
        """Return a live ``Llama`` for *key*, evicting anything else first."""
        with self._lock:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"model for slot '{key}' not found: {path} "
                    f"(run: python tools/download_models.py --only {key})")
            if self._m is not None and (self._key != key or self._path != path):
                self.release()
            if self._m is None:
                t0 = time.perf_counter()
                self._m = self._build(key, path, mmproj, n_ctx, extra or {})
                self._load_seconds = time.perf_counter() - t0
                self._key, self._path = key, path
                self._loaded_at = time.time()
                self._loaded_rss = rss_bytes()
                self.loads += 1
                gc.collect()
                log.info("loaded %s (%s) in %.1fs, rss=%.2f GB",
                         key, path.name, self._load_seconds, rss_gb())
            return self._m

    @staticmethod
    def _build(key: str, path: Path, mmproj: Path | str | None,
               n_ctx: int, extra: dict):
        from llama_cpp import Llama

        kw = dict(model_path=str(path), n_ctx=int(n_ctx),
                  n_threads=CFG.effective_threads, n_batch=CFG.n_batch,
                  verbose=False)
        if CFG.kv_q8:                                   # quantised KV (ADR-002)
            import llama_cpp as _lc
            q8 = getattr(_lc, "GGML_TYPE_Q8_0", None)
            if q8 is not None:                          # int enum, not "q8_0"
                kw.update(type_k=int(q8), type_v=int(q8),
                          # llama.cpp refuses a quantised V cache without FA,
                          # and FA is also ~15% faster on CPU (measured)
                          flash_attn=True)
        if mmproj:
            # vision models need an explicit multimodal chat handler
            from llama_cpp import llama_chat_format
            kw.update(chat_handler=llama_chat_format.Qwen25VLChatHandler(
                clip_model_path=str(mmproj)))
        kw.update(extra)
        return Llama(**kw)

    def release(self) -> int:
        """Drop the heavy model and give the memory back.

        Returns the RSS *after* the drop so tests / the GUI can show that the
        unload is real (TC-UNIT-03: >= 1 GB reclaimed).
        """
        with self._lock:
            if self._m is None:
                return rss_bytes()
            self._m = None
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL(None).malloc_trim(0)        # return pages to the OS
            except Exception:                           # pragma: no cover
                pass
            self._key = None
            self._path = None
            self.evictions += 1
            log.info("released heavy slot, rss=%.2f GB", rss_gb())
            return rss_bytes()

    def __enter__(self):                                # pragma: no cover
        return self

    def __exit__(self, *_):                             # pragma: no cover
        self.release()


HEAVY = ModelSlot()


# ---------------------------------------------------------------------------
# embedder (resident, ~0.15 GB)
# ---------------------------------------------------------------------------
_EMB = None
_EMB_LOCK = threading.Lock()
_EMB_PATH: Path | None = None
EMB_DIM = 384


def embedder() -> object:
    """Lazily construct the resident embedding model (MiniLM-L6 Q4, 384-dim)."""
    global _EMB, _EMB_PATH
    with _EMB_LOCK:
        if _EMB is None:
            from llama_cpp import Llama
            path = CFG.embed_model_path()
            if not path.exists():
                raise FileNotFoundError(
                    f"embedder not found: {path} "
                    "(run: python tools/download_models.py --only embed)")
            _EMB = Llama(model_path=str(path), embedding=True, n_ctx=512,
                         n_threads=CFG.effective_threads, verbose=False,
                         n_batch=256)
            _EMB_PATH = path
            log.info("embedder resident: %s (%.0f MB)", path.name,
                     rss_bytes() / 1e6)
        return _EMB


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Normalised 384-dim vectors, one per input string.

    Handles both pooling modes a GGUF can carry: if llama.cpp reports pooled
    vectors we use them verbatim, otherwise (pooling NONE) we mean-pool the
    token embeddings the way sentence-transformers does.
    """
    if not texts:
        return []
    model = embedder()
    prepared = [(t or " ").strip() or " " for t in texts]
    raw = model.embed(prepared, normalize=False, truncate=True)
    out: list[list[float]] = []
    for item in raw:
        vec = _as_vector(item)
        vec = _l2_normalize(vec)
        out.append(vec)
    return out


def _as_vector(item) -> list[float]:
    """Unify pooled [d] and token-level [tokens][d] embedding outputs."""
    arr = np.asarray(item, dtype=np.float32)
    if arr.ndim == 1:
        return arr.tolist()
    if arr.ndim == 2:
        return arr.mean(axis=0).tolist()                # mean pooling
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0).tolist()


def _l2_normalize(vec: list[float] | np.ndarray) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def embed_dim() -> int:
    return EMB_DIM


def unload_embedder() -> None:
    global _EMB, _EMB_PATH
    with _EMB_LOCK:
        _EMB = None
        _EMB_PATH = None
        gc.collect()


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def _record(result: GenResult) -> GenResult:
    HEAVY.last_gen = result
    result.peak_rss_gb = rss_gb()
    return result


def generate(prompt: str, system: str = "", max_tokens: int | None = None,
             temperature: float = 0.2, stop: Sequence[str] | None = None) -> GenResult:
    """Text-only completion on the heavy slot (LLM role)."""
    m = HEAVY.acquire("llm", CFG.active_llm, n_ctx=CFG.llm_ctx)
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    t0 = time.perf_counter()
    kw = dict(max_tokens=max_tokens or CFG.llm_max_tokens, temperature=temperature)
    if stop:
        kw["stop"] = list(stop)
    r = m.create_chat_completion(msgs, **kw)
    dt = time.perf_counter() - t0
    n = int(r["usage"]["completion_tokens"])
    text = r["choices"][0]["message"]["content"] or ""
    return _record(GenResult(text=text, tokens=n, seconds=dt,
                             tok_per_s=n / dt if dt else 0.0))


def vision_generate(image: "Path | np.ndarray", prompt: str,
                    max_tokens: int | None = None, temperature: float = 0.0,
                    ) -> GenResult:
    """One VLM call: page image in, model text out (ADR-008)."""
    m = HEAVY.acquire("vlm", CFG.model_path("vlm_gguf"),
                      mmproj=CFG.model_path("vlm_mmproj"), n_ctx=CFG.vlm_ctx)
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _image_uri(image)}},
        {"type": "text", "text": prompt}]}]
    t0 = time.perf_counter()
    r = m.create_chat_completion(
        msgs, max_tokens=max_tokens or CFG.runtime_vlm_max_tokens,
        temperature=temperature)
    dt = time.perf_counter() - t0
    n = int(r["usage"]["completion_tokens"])
    text = r["choices"][0]["message"]["content"] or ""
    return _record(GenResult(text=text, tokens=n, seconds=dt,
                             tok_per_s=n / dt if dt else 0.0))


def vision_generate_raw(model, image: "Path | np.ndarray", prompt: str,
                        max_tokens: int, temperature: float = 0.0) -> GenResult:
    """Same call against an *already loaded* model (per-crop fallback loop).

    Re-uses the resident VLM instead of paying a second load/evict cycle.
    """
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _image_uri(image)}},
        {"type": "text", "text": prompt}]}]
    t0 = time.perf_counter()
    r = model.create_chat_completion(msgs, max_tokens=max_tokens,
                                     temperature=temperature)
    dt = time.perf_counter() - t0
    n = int(r["usage"]["completion_tokens"])
    return _record(GenResult(text=r["choices"][0]["message"]["content"] or "",
                             tokens=n, seconds=dt,
                             tok_per_s=n / dt if dt else 0.0))


# ---------------------------------------------------------------------------
# availability / reporting
# ---------------------------------------------------------------------------
def model_report() -> dict:
    """Everything the settings dialog and ``cli bench`` want to show."""
    files = {}
    for key in ("vlm_gguf", "vlm_mmproj", "llm_gguf", "llm_fallback_gguf",
                "embed_gguf", "embed_alt_gguf"):
        p = CFG.model_path(key)
        files[key] = {"path": str(p), "exists": p.exists(),
                      "mb": round(p.stat().st_size / 1e6, 1) if p.exists() else 0}
    return {
        "files": files,
        "threads": CFG.effective_threads,
        "tier": CFG.effective_tier,
        "heavy": {"key": HEAVY.key, "resident_gb": round(HEAVY.resident_gb, 2),
                  "loads": HEAVY.loads, "evictions": HEAVY.evictions,
                  "state": HEAVY.state()},
        "rss_gb": round(rss_gb(), 2),
        "embedder_loaded": _EMB is not None,
    }


def unload_all() -> int:
    """Free every model (used before/after benchmarks and on app exit)."""
    unload_embedder()
    return HEAVY.release()
