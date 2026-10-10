"""Clocks shared by the vision and voice paths.

All timing logic (how long a car has been stopped, when a question ended, alert latency) runs on one clock.
Both clocks use time.perf_counter: on Windows time.monotonic only ticks every 15.6 ms, which would quantize
frame timestamps and the per-second rates computed from them.

- camera mode: `WallClock`: seconds since start on the monotonic clock;
- file mode: `MediaClock`: the video's own timeline, which advances in real time while playing and can be
  paused and seeked. Logic driven by media time is deterministic and survives pause/seek.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol


class Clock(Protocol):
    """A source of 'now' in seconds on the pipeline's timeline."""

    def now(self) -> float:
        """Current time in seconds."""
        ...


class WallClock:
    """Monotonic seconds since construction (camera mode)."""

    def __init__(self, monotonic: Callable[[], float] = time.perf_counter) -> None:
        self._monotonic = monotonic
        self._t0 = monotonic()

    def now(self) -> float:
        """Seconds since this clock was created."""
        return self._monotonic() - self._t0


class MediaClock:
    """Video-timeline clock (file mode): advances at `rate` × real time while playing; pausable and seekable.

    Thread-safe: the capture thread drives pause/seek while other threads read `now()`.
    """

    def __init__(
        self, start: float = 0.0, rate: float = 1.0, monotonic: Callable[[], float] = time.perf_counter
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._rate = rate
        self._media_anchor = start
        self._wall_anchor = monotonic()
        self._paused = True

    @property
    def paused(self) -> bool:
        """True while the clock is stopped."""
        with self._lock:
            return self._paused

    def now(self) -> float:
        """Current media time in seconds."""
        with self._lock:
            return self._now_locked()

    def _now_locked(self) -> float:
        if self._paused:
            return self._media_anchor
        return self._media_anchor + (self._monotonic() - self._wall_anchor) * self._rate

    def play(self) -> None:
        """Start or resume advancing from the current media time."""
        with self._lock:
            if self._paused:
                self._wall_anchor = self._monotonic()
                self._paused = False

    def pause(self) -> None:
        """Freeze the media time."""
        with self._lock:
            if not self._paused:
                self._media_anchor = self._now_locked()
                self._paused = True

    def seek(self, media_t: float) -> None:
        """Jump to `media_t` (clamped at 0); keeps the play/pause state."""
        with self._lock:
            self._media_anchor = max(0.0, media_t)
            self._wall_anchor = self._monotonic()
