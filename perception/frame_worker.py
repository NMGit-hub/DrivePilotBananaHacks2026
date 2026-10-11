"""A model on its own thread, on the newest frame only, a few times a second (the sign detector, the lane model).

Models that cost several milliseconds a frame don't run in the 30 fps perception loop: the world thread hands each
processed frame over (`offer`, a one-slot hand-off that keeps only the newest), and the worker's thread runs the
model at most `fps` times a second and passes (frame, result) to `on_result`. The model is built on that thread by
`factory` (its CUDA state belongs to the thread that made it, and building it verifies the GPU). If it can't be
built, the thread ends and `error` says why; a failure on one frame is logged and the next frame is tried.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Generic, Self, TypeVar

from core.logging_setup import get_logger
from core.types import Frame

log = get_logger(__name__)

R = TypeVar("R")


class FrameWorker(Generic[R]):
    """Runs `factory()`'s model on the newest frame, at most `fps` times a second, on a thread called `name`."""

    def __init__(
        self,
        name: str,
        factory: Callable[[], Callable[[Frame], R]],
        on_result: Callable[[Frame, R], None],
        fps: float,
    ) -> None:
        self.name = name
        self._factory = factory
        self._on_result = on_result
        self._interval = 1.0 / fps
        self._cond = threading.Condition()
        self._pending: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ready = threading.Event()  # set once the model is built (or failed: see `error`)
        self.error: str | None = None
        self.processed = 0
        self.last_ms: float | None = None

    def start(self) -> Self:
        """Start the thread (returns self)."""
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the thread."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        """The thread is alive (False after a failed start)."""
        return self._thread is not None and self._thread.is_alive()

    def offer(self, frame: Frame) -> None:
        """A new frame (any thread): only the newest waits; never blocks."""
        with self._cond:
            self._pending = frame
            self._cond.notify()

    def _run(self) -> None:
        try:
            model = self._factory()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model could not start", worker=self.name)
            self.ready.set()
            return
        self.ready.set()
        log.info("model ready", worker=self.name)
        next_at = 0.0
        while not self._stop.is_set():
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(0.5)
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            delay = next_at - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
                with self._cond:  # a newer frame arrived while we waited: use that one
                    if self._pending is not None:
                        frame, self._pending = self._pending, None
            next_at = time.perf_counter() + self._interval
            started = time.perf_counter()
            try:
                result = model(frame)
            except Exception:
                log.exception("model failed on a frame", worker=self.name, t=frame.t)
                continue
            self.last_ms = (time.perf_counter() - started) * 1000.0
            self.processed += 1
            try:
                self._on_result(frame, result)
            except Exception:
                log.exception("result handling failed", worker=self.name, t=frame.t)
