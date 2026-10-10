"""Scene memory for "what was that sign?" (V2-M4): a frame every `spacing_s` for the last `keep_s`, and the best
close-up of every sign seen in that time.

A frame is kept downscaled to `width` (60 frames of 960 px are ~100 MB); a sign's close-up is cut from the full
frame when its box is the largest seen so far for that track (signs grow as we approach them) and not cut off by
the frame's edge. Signs today are YOLO's stop signs; the V2-M6 sign detector adds its own kinds through
`add_sign`. Fed from the world thread (`add`), read from the voice thread (`frames`, `signs`): thread-safe.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from core.config import MemoryConfig
from core.types import Frame
from perception.detector import Track

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SignCrop:
    """The best close-up of one sign we saw."""

    track_id: int
    kind: str  # "stop sign", or a V2-M6 sign kind ("speed_limit", ...)
    t: float  # when this close-up was taken (media time)
    image: np.ndarray  # BGR close-up (the box padded by half its size)
    box: Box  # in the full frame
    side: str  # "left" | "ahead" | "right" (where it was in the frame)
    text: str | None = None  # what a detector read on it (e.g. "35 mph"), if anything


def sign_crop(image: np.ndarray, box: Box, height: int = 192) -> np.ndarray | None:
    """A close-up of a sign: the box padded by half its size, scaled to `height` pixels."""
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * 0.5, (y2 - y1) * 0.5
    cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx2, cy2 = min(w_img, int(x2 + pad_x)), min(h_img, int(y2 + pad_y))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    scale = height / crop.shape[0]
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), height), interpolation=interpolation)


def _side(box: Box, width: int) -> str:
    center = (box[0] + box[2]) / 2 / max(width, 1)
    return "left" if center < 0.4 else "right" if center > 0.6 else "ahead"


class SceneMemory:
    """The last minute of the drive, for questions about what we just passed."""

    def __init__(self, cfg: MemoryConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._frames: deque[Frame] = deque()
        self._signs: dict[tuple[str, int], SignCrop] = {}

    def reset(self) -> None:
        """Forget everything (a seek or a new segment: older frames are from another moment)."""
        with self._lock:
            self._frames.clear()
            self._signs.clear()

    def add(self, frame: Frame, tracks: Sequence[Track]) -> None:
        """One processed frame (world thread): keep it if it's time, and close-ups of the signs in it."""
        cfg = self.cfg
        with self._lock:
            if self._frames and frame.t < self._frames[-1].t:
                self._frames.clear()
                self._signs.clear()
            keep = not self._frames or frame.t - self._frames[-1].t >= cfg.spacing_s
        if keep:
            small = _downscale(frame, cfg.width)
            with self._lock:
                self._frames.append(small)
                while self._frames and frame.t - self._frames[0].t > cfg.keep_s:
                    self._frames.popleft()
        for tr in tracks:
            if tr.cls in cfg.sign_classes and tr.conf >= cfg.sign_min_conf:
                self.add_sign(frame, tr.track_id, tr.cls, tr.box)
        with self._lock:
            for key in [k for k, s in self._signs.items() if frame.t - s.t > cfg.keep_s]:
                del self._signs[key]

    def add_sign(self, frame: Frame, track_id: int, kind: str, box: Box, text: str | None = None) -> None:
        """A sign seen in `frame`: kept if it is the best (largest, whole) view of it so far."""
        cfg = self.cfg
        x1, y1, x2, y2 = box
        margin = 2.0
        if x1 <= margin or y1 <= margin or x2 >= frame.width - margin or y2 >= frame.height - margin:
            return  # cut off by the frame's edge: a later or earlier whole view is better
        if y2 - y1 < cfg.sign_min_px:
            return
        key = (kind, track_id)
        area = (x2 - x1) * (y2 - y1)

        def better(old: SignCrop | None) -> bool:
            return old is None or text is not None or (old.box[2] - old.box[0]) * (old.box[3] - old.box[1]) < area

        with self._lock:
            if not better(self._signs.get(key)):
                return
        crop = sign_crop(frame.image, box)  # (outside the lock: cutting and scaling takes a moment)
        if crop is None:
            return
        with self._lock:
            old = self._signs.get(key)  # (again: another thread may have kept a better view meanwhile)
            if not better(old):
                return
            keep_text = text if text is not None else (old.text if old is not None else None)
            self._signs[key] = SignCrop(track_id, kind, frame.t, crop, box, _side(box, frame.width), keep_text)

    def frames(self, window_s: float, count: int, now: float | None = None) -> list[Frame]:
        """Up to `count` remembered frames spread over the last `window_s` (newest last)."""
        with self._lock:
            frames = list(self._frames)
        if not frames:
            return []
        end = frames[-1].t if now is None else now
        recent = [f for f in frames if end - f.t <= window_s]
        if len(recent) <= count:
            return recent
        step = (len(recent) - 1) / (count - 1) if count > 1 else 0.0
        picked = [recent[round(i * step)] for i in range(count)] if count > 1 else [recent[-1]]
        return list(dict.fromkeys(picked))  # (no duplicates when rounding meets)

    def signs(self, window_s: float, count: int, now: float) -> list[SignCrop]:
        """The close-ups of signs seen in the last `window_s`, most recent first."""
        with self._lock:
            signs = [s for s in self._signs.values() if now - s.t <= window_s]
        signs.sort(key=lambda s: -s.t)
        return signs[:count]


def _downscale(frame: Frame, width: int) -> Frame:
    """A copy of the frame at most `width` wide (same time, index and capture time)."""
    h, w = frame.image.shape[:2]
    if w <= width:
        return frame
    image = cv2.resize(frame.image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return Frame(image=image, t=frame.t, index=frame.index, captured_at=frame.captured_at, segment=frame.segment)
