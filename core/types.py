"""Small data types shared across packages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One decoded video frame, shared read-only between threads.

    Attributes:
        image: BGR uint8 array (H, W, 3), as OpenCV uses. Made read-only on construction so no consumer
            (UI, vision model, recorder) can draw on a frame others are still using; copy it to modify.
        t: time in seconds on the pipeline clock (video timeline in file mode, wall clock in camera mode).
        index: frame number in the source (monotonic within a segment; gaps mean dropped frames).
        captured_at: `time.perf_counter()` when the frame was decoded/received (for latency measurement;
            not time.monotonic, which only ticks every 15.6 ms on Windows).
        segment: increments on every discontinuity (seek, loop restart); trackers and timers reset on change.

    Equality is identity (eq=False): comparing pixel arrays element-wise is never what callers want.
    """

    image: np.ndarray
    t: float
    index: int
    captured_at: float
    segment: int = 0

    def __post_init__(self) -> None:
        self.image.setflags(write=False)

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.image.shape[0])
