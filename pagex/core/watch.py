"""WP9 helper - GUI-agnostic directory watching.

Kept out of ``gui/`` so the watch logic is unit-testable and so the CLI can
expose ``pagex watch`` too.  The Qt layer only supplies the timer.

Self-ingest protection (pitfall register): temp files (``~$``, ``.part``,
``.tmp``), our own artifact directory, and anything we just wrote are ignored,
otherwise a watch folder that *is* the artifacts folder would loop forever.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import CFG
from .pipeline import IMAGE_SUFFIXES, PDF_SUFFIXES

log = logging.getLogger(__name__)

IGNORE_PREFIXES = ("~$", ".", "#")
IGNORE_SUFFIXES = (".part", ".tmp", ".crdownload", ".download", ".swp")


def is_candidate(path: str | Path, *, watch_root: Path | None = None,
                 artifacts: Path | None = None) -> bool:
    """True when *path* is a file we should ingest (filters applied)."""
    p = Path(path)
    name = p.name
    if name.startswith(IGNORE_PREFIXES):
        return False
    if name.lower().endswith(IGNORE_SUFFIXES):
        return False
    suffix = p.suffix.lower()
    if suffix not in IMAGE_SUFFIXES | PDF_SUFFIXES:
        return False
    artifacts = artifacts or CFG.artifacts_dir
    try:
        resolved = p.resolve()
        if resolved.is_relative_to(artifacts.resolve()):
            return False
    except Exception:                                   # pragma: no cover
        pass
    if watch_root is not None:
        try:
            if not resolved.is_relative_to(Path(watch_root).resolve()):
                return False
        except Exception:
            return False
    return True


def pending(paths: Iterable[str | Path], *, watch_root: Path | None = None,
            seen: set | None = None, stable_for: float = 0.0) -> list[Path]:
    """Filter *paths* down to new/stable candidates.

    *stable_for* is the copy-in-progress guard: a file still being written has
    a changing mtime/size, so we wait until it has been quiet for that long.
    """
    out: list[Path] = []
    seen = seen if seen is not None else set()
    for raw in paths:
        p = Path(raw)
        if not p.exists() or not is_candidate(p, watch_root=watch_root):
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        if stable_for > 0 and not _stable(p, stable_for):
            continue
        out.append(p)
    return out


def _stable(path: Path, seconds: float) -> bool:
    try:
        st1 = path.stat()
        time.sleep(min(seconds, 0.05)) if seconds > 0.05 else time.sleep(seconds)
        st2 = path.stat()
        return st1.st_size == st2.st_size and st1.st_mtime == st2.st_mtime
    except OSError:                                     # pragma: no cover
        return False


def scan(dir_path: str | Path) -> list[Path]:
    """All ingestible files in a directory (non-recursive)."""
    d = Path(dir_path)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and is_candidate(p, watch_root=d))


class WatchState:
    """Debounced queue of files seen since the last drain.

    The Qt layer calls :meth:`on_change` from a ``QFileSystemWatcher`` signal,
    starts (or restarts) a ``QTimer`` of ``debounce_ms``, and on timeout calls
    :meth:`drain` to get the files that survived the debounce window.
    """

    def __init__(self, watch_dir: str | Path | None = None,
                 debounce_s: float = 3.0, ignore_seen: bool = True):
        self.watch_dir = Path(watch_dir) if watch_dir else CFG.watch_dir
        self.debounce_s = debounce_s
        self.ignore_seen = ignore_seen
        self._pending: dict[str, float] = {}
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    # -- Qt signal target ------------------------------------------------
    def on_change(self, path: str | None = None) -> None:
        """Record a filesystem event (safe to call from any thread)."""
        with self._lock:
            if path is None:
                if self.watch_dir:
                    for p in scan(self.watch_dir):
                        self._pending[str(p)] = time.time()
                return
            p = Path(path)
            if not is_candidate(p, watch_root=self.watch_dir):
                return
            self._pending[str(p)] = time.time()

    def drain(self, now: float | None = None) -> list[Path]:
        """Return files whose events are older than the debounce window."""
        now = now if now is not None else time.time()
        ready: list[Path] = []
        with self._lock:
            for key in list(self._pending):
                if now - self._pending[key] < self.debounce_s:
                    continue
                p = Path(key)
                if not p.exists():
                    self._pending.pop(key, None)
                    continue
                if self.ignore_seen and key in self._seen:
                    self._pending.pop(key, None)
                    continue
                self._seen.add(key)
                self._pending.pop(key, None)
                ready.append(p)
        return sorted(ready)

    def drain_now(self) -> list[Path]:
        with self._lock:
            keys = list(self._pending)
            self._pending.clear()
            out = []
            for key in keys:
                p = Path(key)
                if p.exists() and not (self.ignore_seen and key in self._seen):
                    self._seen.add(key)
                    out.append(p)
        return sorted(out)

    @property
    def seen(self) -> set[str]:
        return set(self._seen)

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._seen.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return {"dir": str(self.watch_dir or ""), "pending": len(self._pending),
                    "seen": len(self._seen), "debounce_s": self.debounce_s}
