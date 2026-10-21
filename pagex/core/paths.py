"""Cross-platform path resolution - one place, four modes (Windows + Linux).

Every relative path in ``pagex.toml`` (``db``, ``artifacts``, ``models``, ...)
and every "where does my config live" question is answered here, so the rest
of the codebase never calls ``Path(__file__)`` or hardcodes ``/``.

Resolution ladder (highest priority first):

1. **``PAGEX_HOME``** - portable mode: config + db + artifacts + models all
   live under that single directory.  This is what the USB/offline demo pack
   sets so the whole app runs from a folder with zero installation.
2. **source checkout** - the git repository root (``pyproject.toml`` sits next
   to it): ``./pagex.toml``, ``./data``, ``./models``.  Keeps the developer
   workflow and the test suite exactly as they were.
3. **frozen build** (PyInstaller: ``sys.frozen``) - no ``__file__``-relative
   data at all: user data comes from platformdirs, bundled resources come
   from the executable directory (``sys._MEIPASS`` / next to the exe).
4. **installed package** (``pip install pagex``) - platformdirs locations:

   ================  ==========================================
   OS                config                     data
   ================  ==========================================
   Windows           ``%APPDATA%\\pagex\\``     ``%LOCALAPPDATA%\\pagex\\``
   Linux             ``~/.config/pagex/``       ``~/.local/share/pagex/``
   ================  ==========================================

``platformdirs`` is used when installed; a stdlib fallback with identical
semantics is used otherwise (a bare Python must never fail to boot).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

APP_NAME = "pagex"

try:                                   # pragma: no cover - trivial guard
    from platformdirs import user_config_dir as _pf_user_config_dir
    from platformdirs import user_data_dir as _pf_user_data_dir
    _HAVE_PLATFORMDIRS = True
except Exception:                      # pragma: no cover
    _HAVE_PLATFORMDIRS = False


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def is_frozen() -> bool:
    """True inside a PyInstaller (or similar) frozen executable."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Directory bundled data files are extracted to / live next to.

    Frozen onedir/onefile builds put ``datas`` under ``sys._MEIPASS``;
    plain Python resolves to the repository/package root that makes sense for
    the current mode (never used for *writable* data - see :func:`base_dir`).
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(sys.executable).resolve().parent if is_frozen() else repo_root() or Path.cwd()


def repo_root() -> Path | None:
    """Git/source checkout root, or ``None`` when installed or frozen."""
    if is_frozen():
        return None
    here = Path(__file__).resolve()
    for parent in here.parents:
        if not ((parent / "pyproject.toml").is_file() and (parent / "pagex").is_dir()):
            continue
        # never mistake a site-packages/... layout for a checkout
        if (parent / ".git").exists() or (parent / "tests").is_dir():
            return parent
    return None


def pagex_home() -> Path | None:
    """``PAGEX_HOME`` portable override (expanded, absolute)."""
    raw = os.environ.get("PAGEX_HOME", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def refresh() -> None:
    """Forget memoised answers (tests, or ``PAGEX_HOME`` changed at runtime)."""
    default_config_path.cache_clear()
    base_dir.cache_clear()


def user_config_dir() -> Path:
    if _HAVE_PLATFORMDIRS:             # pragma: no cover - trivial
        # Windows config is roaming; data (database/artifacts/models) is local.
        return Path(_pf_user_config_dir(APP_NAME, appauthor=False, roaming=True))
    if os.name == "nt":                # pragma: no cover
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_NAME


def user_data_dir() -> Path:
    if _HAVE_PLATFORMDIRS:             # pragma: no cover - trivial
        return Path(_pf_user_data_dir(APP_NAME, appauthor=False))
    if os.name == "nt":                # pragma: no cover
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


# ---------------------------------------------------------------------------
# the two questions everything else derives from
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def default_config_path() -> Path:
    """Where ``pagex.toml`` is read from (and written back to)."""
    explicit = os.environ.get("PAGEX_CONFIG", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    home = pagex_home()
    if home is not None:
        return home / "pagex.toml"
    root = repo_root()
    if root is not None:
        return root / "pagex.toml"
    return user_config_dir() / "pagex.toml"


@lru_cache(maxsize=None)
def base_dir() -> Path:
    """Directory that relative ``pagex.toml`` paths resolve against.

    Writable state (db, artifacts, models, bench.json) - never the bundle dir.
    """
    home = pagex_home()
    if home is not None:
        return home
    root = repo_root()
    if root is not None:
        return root
    if is_frozen():
        # frozen without PAGEX_HOME: still fully portable on disk (user data),
        # while bundled read-only resources live in bundle_dir().
        return user_data_dir()
    return user_data_dir()


def data_dir() -> Path:
    """Alias of :func:`base_dir` for call sites that read better by name."""
    return base_dir()


def bench_path() -> Path:
    """``bench.json`` (hardware probe output) - beside the active config."""
    return base_dir() / "bench.json"


def logs_dir() -> Path:
    d = base_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# bundled (read-only) resources
# ---------------------------------------------------------------------------
def samples_dir(create: bool = False) -> Path:
    """Best location for the sample corpus (read-only copies or generated).

    Order: bundled with the app -> repository checkout -> writable data dir
    (``pagex demo`` generates the synthetic samples there when missing).
    """
    candidates: list[Path] = []
    home = pagex_home()
    if home is not None:
        candidates.append(home / "samples")
    bundle = bundle_dir() / "samples"
    candidates.append(bundle)
    root = repo_root()
    if root is not None:
        candidates.append(root / "samples")
    if is_frozen():
        candidates.append(Path(sys.executable).resolve().parent / "samples")
    candidates.append(base_dir() / "samples")

    for cand in candidates:
        if cand.exists():
            return cand
    target = candidates[0] if home is not None else (
        bundle if bundle.parent.exists() and is_frozen() else candidates[-1])
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def resource(*parts: str) -> Path | None:
    """First existing path among the bundled candidates for *parts*."""
    for cand in (bundle_dir(), *( [repo_root()] if repo_root() else [] )):
        p = cand.joinpath(*parts)
        if p.exists():
            return p
    return None


def describe() -> dict[str, str]:
    """Human-readable mode report (``pagex doctor`` / bug reports)."""
    return {
        "mode": ("portable (PAGEX_HOME)" if pagex_home() else
                 "source checkout" if repo_root() else
                 "frozen" if is_frozen() else "installed"),
        "frozen": str(is_frozen()),
        "config": str(default_config_path()),
        "base_dir": str(base_dir()),
        "samples": str(samples_dir()),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "platformdirs": str(_HAVE_PLATFORMDIRS),
    }
