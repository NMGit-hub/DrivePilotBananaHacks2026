"""The sign detector's own thread (V2-M6): the latest frame, a few times a second, off the perception thread.

The detector (perception/signs.py, 736x1280 TensorRT) costs several milliseconds a frame, so it runs on a
`FrameWorker` (perception/frame_worker.py): only the newest frame, at most `fps` times a second, the detector built
on that thread.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from core.types import Frame
from perception.frame_worker import FrameWorker


class FrameDetector(Protocol):
    """What the worker needs from a detector (perception.signs.SignDetector)."""

    def detect(self, frame: Frame) -> Sequence[Any]: ...


class SignWorker(FrameWorker[Sequence[Any]]):
    """Runs a sign detector on the newest frame, at most `fps` times a second (thread "signs")."""

    def __init__(
        self,
        factory: Callable[[], FrameDetector],
        on_result: Callable[[Frame, Sequence[Any]], None],
        fps: float,
    ) -> None:
        super().__init__("signs", lambda: factory().detect, on_result, fps)
