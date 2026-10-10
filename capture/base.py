"""The video-source interface shared by the camera and file inputs.

A source runs its own capture thread and keeps only the newest decoded frame ("latest-frame" hand-off), so a
slow consumer never builds up latency: frames it did not take in time are dropped and counted. File sources
can instead run in lock-step mode (`realtime: false`) where every frame is delivered, for benchmarks and evals.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.clock import Clock
from core.logging_setup import get_logger
from core.types import Frame

log = get_logger(__name__)


@dataclass(slots=True)
class SourceStats:
    """Counters for one source run."""

    decoded: int = 0  # frames produced by the capture thread
    delivered: int = 0  # frames handed to the consumer
    dropped: int = 0  # frames replaced before the consumer took them (or skipped to keep real time)


class VideoSource(ABC):
    """Base class: a capture thread plus a one-slot latest-frame buffer.

    Subclasses implement `_run()` (the capture loop) and call `_publish(frame)` for each frame and
    `_finish()` when the stream ends. Consumers call `read()`.
    """

    def __init__(self, clock: Clock, name: str) -> None:
        self.clock = clock
        self.name = name
        self.stats = SourceStats()
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._finished = False
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lockstep = False  # True: _publish blocks until the consumer took the previous frame

    # ----------------------------------------------------------------------------- public interface

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for a live camera (no pause/seek), False for a recorded file."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frame rate of the source."""

    def start(self) -> None:
        """Open the source and start the capture thread."""
        if self._thread is not None:
            raise RuntimeError(f"{self.name} already started")
        try:
            self._open()
        except BaseException:
            self._close()  # release whatever _open() acquired before failing
            raise
        self._thread = threading.Thread(target=self._thread_main, name=f"capture-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture thread and release the device/file. Safe to call more than once."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        self._close()

    def read(self, timeout: float | None = None) -> Frame | None:
        """Return the next unseen frame, waiting up to `timeout` seconds.

        Returns None on timeout or when the stream has finished. Raises the capture thread's error if it failed.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._latest is None:
                if self._error is not None:
                    raise RuntimeError(f"{self.name} capture failed") from self._error
                if self._finished or self._stop.is_set():
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            frame, self._latest = self._latest, None
            self.stats.delivered += 1
            self._cond.notify_all()
            return frame

    @property
    def finished(self) -> bool:
        """True once the stream has ended and the last frame was consumed."""
        with self._cond:
            return self._finished and self._latest is None

    # Playback controls: meaningful for files; live sources ignore them.
    def toggle_pause(self) -> bool:
        """Pause or resume playback. Returns the new paused state (always False for live sources)."""
        return False

    def seek_relative(self, delta_s: float) -> None:  # noqa: B027 - optional hook, default no-op for live sources
        """Skip forward/backward by `delta_s` seconds (no-op for live sources)."""

    # ------------------------------------------------------------------------------- for subclasses

    @abstractmethod
    def _open(self) -> None:
        """Open the device/file (called from start(), in the caller's thread, so errors surface immediately)."""

    @abstractmethod
    def _run(self) -> None:
        """Capture loop; return when the stream ends or `self._stop` is set."""

    @abstractmethod
    def _close(self) -> None:
        """Release resources."""

    def _thread_main(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            log.exception("capture thread crashed", source=self.name)
            with self._cond:
                self._error = exc
                self._cond.notify_all()
        finally:
            self._finish()

    def _publish(self, frame: Frame) -> None:
        """Hand a frame to the consumer (replacing an unconsumed one unless in lock-step mode)."""
        with self._cond:
            self.stats.decoded += 1
            if self._lockstep:
                while self._latest is not None and not self._stop.is_set():
                    self._cond.wait(0.1)
            elif self._latest is not None:
                self.stats.dropped += 1
            self._latest = frame
            self._cond.notify_all()

    def _finish(self) -> None:
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def _count_skipped(self, n: int = 1) -> None:
        """Count frames skipped without decoding (e.g. to catch up with real time)."""
        with self._cond:
            self.stats.dropped += n
