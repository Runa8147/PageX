"""Runtime configuration for PageX.

ADR: one flat ``Config`` object (``CFG``) loaded from ``pagex.toml``.  Flat
on purpose - every subsystem does ``from .config import CFG`` and reads
``CFG.llm_gguf`` etc.  The GUI settings dialog calls :meth:`Config.reload`
/ :meth:`Config.save` to persist changes back to the TOML file.

Cross-platform: *where* ``pagex.toml`` lives and what relative paths inside
it resolve against is decided by :mod:`pagex.core.paths` (``PAGEX_HOME``
portable mode, source checkout, frozen build, ``pip install``).  Nothing in
this file touches ``__file__``-relative data.
"""
from __future__ import annotations

import logging
import os
import threading
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

#: where relative paths resolve against (dev: repository root)
ROOT = paths.base_dir()
DEFAULT_PATH = paths.default_config_path()
BENCH_PATH = paths.bench_path()

_TRUE = {"1", "true", "yes", "on"}
_NON_TOML_FIELDS = {"path", "base"}


def _base_for(config_path: Path) -> Path:
    """Directory relative paths in *config_path* resolve against."""
    if config_path == paths.default_config_path():
        return paths.base_dir()
    return config_path.parent


@dataclass
class Config:
    # ---- [paths] -------------------------------------------------------
    db: str = "data/pagex.db"
    artifacts: str = "data/artifacts"
    models: str = "models"
    watch: str = ""

    # ---- [models] ------------------------------------------------------
    vlm_gguf: str = "models/Qwen3-VL-2B-Instruct-Q4_K_M.gguf"
    vlm_mmproj: str = "models/mmproj-Qwen3-VL-2B-Instruct-F16.gguf"
    llm_gguf: str = "models/granite-3.3-2b-instruct-Q4_K_M.gguf"
    llm_fallback_gguf: str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    embed_gguf: str = "models/all-minilm-l6-v2-q4.gguf"
    embed_alt_gguf: str = "models/bge-small-en-v1.5-q8_0.gguf"

    # ---- [runtime] -----------------------------------------------------
    n_threads: int = 0          # 0 -> ask bench.json / cpu_count
    vlm_ctx: int = 8192
    llm_ctx: int = 4096
    n_batch: int = 512
    kv_q8: bool = True
    vlm_max_side: int = 1280
    vlm_max_tokens: int = 3072
    llm_max_tokens: int = 1024
    tier: str = ""              # "" = auto (from bench.json)

    # ---- [quality] -----------------------------------------------------
    rrf_k: int = 60
    top_k: int = 5
    chunk_tokens: int = 400
    nms_iou: float = 0.5
    min_box_frac: float = 0.01

    # bookkeeping
    path: Path = field(default=DEFAULT_PATH, repr=False)
    base: Path = field(default=ROOT, repr=False)

    # ------------------------------------------------------------------
    # path helpers (all pagex.toml-relative inputs are resolved here so
    # the app works no matter what the process CWD or OS is)
    # ------------------------------------------------------------------
    def _abs(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.base / p)

    @property
    def db_path(self) -> Path:
        return self._abs(self.db)

    @property
    def artifacts_dir(self) -> Path:
        return self._abs(self.artifacts)

    @property
    def models_dir(self) -> Path:
        return self._abs(self.models)

    @property
    def watch_dir(self) -> Path | None:
        return self._abs(self.watch) if self.watch else None

    def model_path(self, key: str) -> Path:
        return self._abs(getattr(self, key))

    # ------------------------------------------------------------------
    # load / save
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        """Read *path* (default: the platform-aware ``pagex.toml``).

        Relative paths inside the file resolve against :func:`pagex.core.paths.base_dir`
        for the default file (portable / checkout / installed) and against the
        file's own directory for an explicitly supplied config.
        """
        target = (Path(path).expanduser().resolve() if path
                  else paths.default_config_path())
        cfg = cls(path=target, base=_base_for(target))
        if cfg.path.exists():
            with open(cfg.path, "rb") as fh:
                data = tomllib.load(fh)
            flat: dict[str, object] = {}
            for section in data.values():
                if isinstance(section, dict):
                    flat.update(section)
            known = {f.name for f in fields(cls)} - _NON_TOML_FIELDS
            for key, value in flat.items():
                if key not in known:
                    log.debug("pagex.toml: ignoring unknown key %r", key)
                    continue
                default = getattr(cls, key)
                if isinstance(default, bool):
                    value = value if isinstance(value, bool) \
                        else str(value).strip().lower() in _TRUE
                elif isinstance(default, int) and not isinstance(default, bool):
                    value = int(value)
                elif isinstance(default, float):
                    value = float(value)
                else:
                    value = str(value)
                setattr(cfg, key, value)
        else:
            log.debug("no pagex.toml at %s - using defaults", cfg.path)
        return cfg

    def save(self, path: Path | str | None = None) -> Path:
        """Persist the current values back to TOML (settings dialog)."""
        target = Path(path) if path else self.path
        order = {
            "paths": ["db", "artifacts", "models", "watch"],
            "models": ["vlm_gguf", "vlm_mmproj", "llm_gguf",
                       "llm_fallback_gguf", "embed_gguf", "embed_alt_gguf"],
            "runtime": ["n_threads", "vlm_ctx", "llm_ctx", "n_batch", "kv_q8",
                        "vlm_max_side", "vlm_max_tokens", "llm_max_tokens",
                        "tier"],
            "quality": ["rrf_k", "top_k", "chunk_tokens", "nms_iou",
                        "min_box_frac"],
        }
        lines: list[str] = ["# PageX runtime configuration.",
                           "# Regenerated by the GUI settings dialog.", ""]
        for section, keys in order.items():
            lines.append(f"[{section}]")
            for key in keys:
                value = getattr(self, key)
                if isinstance(value, bool):
                    text = "true" if value else "false"
                elif isinstance(value, float):
                    text = repr(value)
                else:
                    text = f'"{value}"' if isinstance(value, str) else str(value)
                lines.append(f"{key:<14} = {text}")
            lines.append("")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        self.path = target
        log.info("wrote %s", target)
        return target

    # ------------------------------------------------------------------
    # derived values
    # ------------------------------------------------------------------
    @property
    def effective_threads(self) -> int:
        """``n_threads`` resolved from bench.json when set to 0."""
        if self.n_threads and self.n_threads > 0:
            return int(self.n_threads)
        bench = self.read_bench()
        if bench:
            for key in ("n_threads", "threads"):
                if bench.get(key):
                    return int(bench[key])
        return max(1, (os.cpu_count() or 2) // 2)

    @property
    def effective_tier(self) -> str:
        if self.tier in ("1", "2", "3"):
            return str(self.tier)
        bench = self.read_bench()
        tier = str((bench or {}).get("tier", "1"))
        return tier if tier in ("1", "2", "3") else "1"

    # quality knobs respect the degradation ladder (Tier-2 shrinks the page)
    @property
    def runtime_vlm_max_side(self) -> int:
        return 768 if self.effective_tier in ("2", "3") else self.vlm_max_side

    @property
    def runtime_vlm_max_tokens(self) -> int:
        return 2048 if self.effective_tier == "2" else self.vlm_max_tokens

    @property
    def active_llm(self) -> Path:
        """Fall back to the 1.5B chat model only when the chat model itself
        is slow (measured ``bench.llm.tok_per_s`` < 6, ADR ladder).

        Each component degrades on its own measured capability: a slow VLM
        must not force a worse answer model when the primary already meets
        the tier-1 chat threshold.
        """
        primary = self.model_path("llm_gguf")
        alt = self.model_path("llm_fallback_gguf")
        if not primary.exists():
            return alt if alt.exists() else primary
        bench = self.read_bench()
        llm_s = float((bench.get("llm") or {}).get("tok_per_s") or 0.0)
        if bench and 0.0 < llm_s < 6.0 and alt.exists():
            return alt
        return primary

    @property
    def bench_path(self) -> Path:
        """``bench.json`` written by ``pagex bench`` (beside this config)."""
        return self.base / "bench.json"

    def read_bench(self) -> dict:
        """Hardware probe output (empty dict when not probed yet)."""
        import json
        try:
            return json.loads(self.bench_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def missing_models(self) -> list[str]:
        needed = ["vlm_gguf", "vlm_mmproj", "llm_gguf", "embed_gguf"]
        return [k for k in needed if not self.model_path(k).exists()]

    def embed_model_path(self) -> Path:
        primary = self.model_path("embed_gguf")
        if primary.exists():
            return primary
        alt = self.model_path("embed_alt_gguf")
        if alt.exists():
            return alt
        return primary


def _read_bench(cfg: "Config | None" = None) -> dict:
    """``bench.json`` written by ``pagex bench`` / tools/probe.py (may not exist)."""
    return (cfg or CFG).read_bench()


# ---------------------------------------------------------------------------
# process-wide singleton
# ---------------------------------------------------------------------------
CFG = Config.load()


def load_config(path: Path | str | None = None) -> Config:
    """(Re)load ``pagex.toml`` into the process-wide :data:`CFG`.

    The singleton is refreshed **in place** so every module that already did
    ``from .config import CFG`` (store, pipeline, rag, ...) sees the new
    values - both the GUI settings dialog and ``pagex --config`` rely on it.
    """
    global CFG
    fresh = Config.load(path)
    for f in fields(Config):
        if f.name in _NON_TOML_FIELDS:
            continue
        setattr(CFG, f.name, getattr(fresh, f.name))
    CFG.path, CFG.base = fresh.path, fresh.base
    return CFG


def setup_logging(level: int = logging.INFO, logfile: Path | str | None = None) -> None:
    """Console + optional file logging (used by cli.py and the GUI)."""
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)


_lock = threading.Lock()  # re-exported for subsystems that want it
