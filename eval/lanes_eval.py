"""Lane evaluation helpers (V2-M5): cached YOLOPv2 masks, the virtual-drift warp, replays through the LaneTracker.

    build_mask_cache(model, clip, out, fps)   # run the lane model over a clip at `fps`, masks to a .npz
    masks = load_mask_cache(out)              # list[LaneMasks]
    warp_lateral(image, shift_m, ...)         # the frame as seen from a camera `shift_m` to the side

Masks are stored bit-packed (360x640 → 28.8 kB each), so a 40 s clip at 12 fps is ~30 MB.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.config import LanesConfig
from perception.lanes import MASK_SIZE, LaneMasks
from state.lanes import DriftEvent, LaneMeasurement, LanesSnapshot, LaneTracker

CACHE_VERSION = 1


def iter_frames(clip: Path, fps: float, start_s: float = 0.0) -> Iterator[tuple[int, float, np.ndarray]]:
    """(frame index, media time, BGR image) of the clip, sampled at `fps` on the media clock."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {clip}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    next_t = start_s
    index = -1
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                return
            index += 1
            t = index / src_fps
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / fps
            yield index, t, image
    finally:
        cap.release()


def build_mask_cache(
    model: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, dict[str, float]]],
    clip: Path,
    out: Path,
    fps: float,
    transform: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> int:
    """Run `model` (LaneModel.infer) over the clip at `fps` and save the masks; returns the number of frames.

    `transform(t, image)` changes each frame before the model sees it (the virtual-drift warp).
    """
    times, indices, lines, drivable, ms = [], [], [], [], []
    size = (0, 0)
    for index, t, image in iter_frames(clip, fps):
        if transform is not None:
            image = transform(t, image)
        size = (image.shape[1], image.shape[0])
        ll, da, timings = model(image)
        times.append(t)
        indices.append(index)
        lines.append(np.packbits(ll.astype(bool), axis=1))
        drivable.append(np.packbits(da.astype(bool), axis=1))
        ms.append(timings["total"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        version=CACHE_VERSION,
        t=np.array(times),
        index=np.array(indices),
        size=np.array(size),
        lines=np.array(lines),
        drivable=np.array(drivable),
        ms=np.array(ms),
    )
    return len(times)


def load_mask_cache(path: Path) -> list[LaneMasks]:
    """The cached masks of one clip, in time order."""
    data = np.load(path)
    if int(data["version"]) != CACHE_VERSION:
        raise ValueError(f"{path}: cache version {int(data['version'])}, expected {CACHE_VERSION}")
    w, h = (int(v) for v in data["size"])
    out = []
    for t, index, ll, da, ms in zip(data["t"], data["index"], data["lines"], data["drivable"], data["ms"], strict=True):
        out.append(
            LaneMasks(
                t=float(t),
                frame_index=int(index),
                frame_size=(w, h),
                lines=np.unpackbits(ll, axis=1, count=MASK_SIZE[0]),
                drivable=np.unpackbits(da, axis=1, count=MASK_SIZE[0]),
                timings_ms={"total": float(ms)},
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """The tracker's state after one cached frame."""

    t: float
    snapshot: LanesSnapshot
    measurement: LaneMeasurement | None
    events: tuple[DriftEvent, ...]
    speed_mps: float | None


def replay(
    cfg: LanesConfig, masks: list[LaneMasks], hfov_deg: float, speed_at: Callable[[float], float | None]
) -> list[ReplayStep]:
    """Feed a clip's cached masks through a fresh LaneTracker, with GPS speed from `speed_at(t)`."""
    tracker = LaneTracker(cfg, hfov_deg)
    steps = []
    for m in masks:
        speed = speed_at(m.t)
        events = tracker.update(m.t, m, speed)
        steps.append(ReplayStep(m.t, tracker.snapshot(), tracker.last, tuple(events), speed))
    return steps


def at_time(steps: list[ReplayStep], t: float) -> ReplayStep:
    """The last step at or before `t` (the state a question asked at `t` would see)."""
    before = [s for s in steps if s.t <= t + 1e-6]
    return before[-1] if before else steps[0]


@dataclass(slots=True)
class IndexScore:
    """Lane index / count agreement with hand labels."""

    frames: int = 0
    index_answered: int = 0
    index_right: int = 0
    count_labeled: int = 0
    count_answered: int = 0
    count_right: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, clip: str, t: float, label: dict[str, Any], snap: LanesSnapshot) -> None:
        """Score one labeled frame against the tracker's snapshot at that time."""
        self.frames += 1
        idx, count = snap.lane_index, snap.lane_count
        if idx is not None:
            self.index_answered += 1
            self.index_right += int(idx == label["index"])
        if label.get("count") is not None:
            self.count_labeled += 1
            if count is not None:
                self.count_answered += 1
                self.count_right += int(count == label["count"])
        self.rows.append(
            {
                "clip": clip,
                "t": t,
                "index": label["index"],
                "count": label.get("count"),
                "got_index": idx,
                "got_count": count,
                "lanes_right": snap.lanes_right,
                "confidence": snap.confidence,
            }
        )


def speed_from_gpx(path: Path | None) -> Callable[[float], float | None]:
    """GPS speed at video time t from a clip's GPX (None without a GPX or outside it)."""
    if path is None or not path.exists():
        return lambda t: None
    from nav.gps import GpxReplay

    track = GpxReplay(path)

    def speed_at(t: float) -> float | None:
        fix = track.fix_at(t)
        return None if fix is None else fix.speed_mps

    return speed_at


def warp_lateral(image: np.ndarray, shift_m: float, horizon: float, camera_height_m: float) -> np.ndarray:
    """The frame as a camera `shift_m` to the right (left: negative) would see the flat road.

    Moving a camera sideways by d on a flat road moves each ground point at image row y by
    -d * (y - y_horizon) / camera_height pixels (a shear about the horizon); everything is warped with that
    shear (the sky shears the other way, which the lane model doesn't look at).
    """
    h, w = image.shape[:2]
    y_h = horizon * h
    k = -shift_m / camera_height_m
    matrix = np.array([[1.0, k, -k * y_h], [0.0, 1.0, 0.0]], dtype=np.float64)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
