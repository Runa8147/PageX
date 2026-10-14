"""WP9 - background workers.

Hard rule: **the UI thread never infers.**  Every model call, ingest and
export runs on ``QThreadPool.globalInstance()`` and reports back through
Qt signals, so the window stays responsive on a 4-core i3.
"""
from __future__ import annotations

import logging
import traceback
from typing import Callable

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

log = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Marshalled back onto the GUI thread by Qt's signal loop."""
    progress = pyqtSignal(str, int, int)      # stage, current, total
    finished = pyqtSignal(object)             # result payload
    error = pyqtSignal(str)                   # message (traceback in logs)


class FnWorker(QRunnable):
    """Run ``fn(*args, **kwargs)`` off-thread; one worker per job."""

    def __init__(self, fn: Callable, *args, on_finished: Callable | None = None,
                 on_error: Callable | None = None,
                 on_progress: Callable | None = None, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()
        if on_finished:
            self.signals.finished.connect(on_finished)
        if on_error:
            self.signals.error.connect(on_error)
        if on_progress:
            self.signals.progress.connect(on_progress)
        self.setAutoDelete(True)

    # -- progress adapter ------------------------------------------------
    def _progress_adapter(self):
        def emit(stage: str, cur: int, total: int) -> None:
            self.signals.progress.emit(str(stage), int(cur), int(total))
        return emit

    def run(self) -> None:                      # executed on a pool thread
        try:
            code = getattr(getattr(self.fn, "__code__", None), "co_varnames", ())
            if "progress" in code and "progress" not in self.kwargs:
                self.kwargs["progress"] = self._progress_adapter()
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:                # noqa: BLE001 - surface to UI
            log.error("worker %s failed: %s", getattr(self.fn, "__name__", self.fn),
                      exc)
            log.debug(traceback.format_exc())
            self._emit(self.signals.error, str(exc))
        else:
            self._emit(self.signals.finished, result)

    @staticmethod
    def _emit(signal, payload) -> None:
        """Emit defensively: the window may be closing while we finish."""
        try:
            signal.emit(payload)
        except RuntimeError:                     # deleted wrapped C++ object
            log.debug("emit dropped - receiver already destroyed")


def run_in_background(fn: Callable, *args,
                      on_finished: Callable | None = None,
                      on_error: Callable | None = None,
                      on_progress: Callable | None = None,
                      **kwargs) -> FnWorker:
    """Schedule *fn* on the shared thread pool and return the worker."""
    worker = FnWorker(fn, *args, on_finished=on_finished, on_error=on_error,
                      on_progress=on_progress, **kwargs)
    QThreadPool.globalInstance().start(worker)
    return worker


def active_thread_count() -> int:
    return QThreadPool.globalInstance().activeThreadCount()


def wait_all(ms: int = -1) -> bool:
    """Block until the pool drains (used by tests and app shutdown)."""
    return QThreadPool.globalInstance().waitForDone(ms)
