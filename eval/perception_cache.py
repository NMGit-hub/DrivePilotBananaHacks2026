"""Perception results of a clip cached on disk, so logic downstream of perception (parking, pedestrians, lanes) can
be replayed and tuned in seconds instead of re-running the GPU each time.

    build_cache(cfg, clip, out)   # TensorRT YOLO26s + ByteTrack + ego motion, every frame (lock-step)
    frames = load_cache(out)      # list[CachedFrame]

A cached frame keeps what the world state reads from a `PerceptionResult`: the time, frame size and index, the
tracks and the ego-motion verdict with its homography. The image itself is not kept (decode it again with
`read_frame` when a check needs it). The cache records the perception settings it was built with; `load_cache`
refuses a cache built with different ones.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.config import AppConfig
from perception.detector import Track

CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CachedFrame:
    """What downstream logic needs from one perception result."""

    index: int  # frame number in the clip (0-based)
    t: float  # media time (s)
    width: int
    height: int
    tracks: tuple[Track, ...]
    ego_stopped: bool
    homography: np.ndarray | None


def settings_key(cfg: AppConfig) -> str:
    """A short hash of the settings that change perception's output (detector, tracker, ego motion)."""
    data = {"perception": cfg.perception.model_dump(mode="json"), "ego": cfg.ego_motion.model_dump(mode="json")}
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def build_cache(
    cfg: AppConfig,
    clip: Path,
    out: Path,
    max_s: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> int:
    """Run perception lock-step over the clip (or its first `max_s` seconds: `scripts/align_gpx.py` on a long
    drive) and write the cache; returns the number of frames. `progress(t)` is called every 10 s of video."""
    from capture.file_source import FileSource
    from perception.detector import Detector
    from perception.ego_motion import EgoMotionEstimator
    from perception.pipeline import PerceptionResult, PerceptionWorker

    detector = Detector(cfg.perception, cfg.root)
    source = FileSource(clip, realtime=False)
    worker = PerceptionWorker(source, detector, EgoMotionEstimator(cfg.ego_motion))
    frames: list[CachedFrame] = []
    enough = threading.Event()
    reported = [0.0]

    def listener(result: PerceptionResult) -> None:
        if max_s is not None and result.frame.t > max_s:
            enough.set()
            return
        if progress is not None and result.frame.t - reported[0] >= 10.0:
            reported[0] = result.frame.t
            progress(result.frame.t)
        h = result.ego.homography
        frames.append(
            CachedFrame(
                index=len(frames),
                t=result.frame.t,
                width=result.frame.width,
                height=result.frame.height,
                tracks=tuple(result.tracks),
                ego_stopped=result.ego.stopped,
                homography=None if h is None else np.array(h, dtype=np.float64),
            )
        )

    worker.add_listener(listener)
    source.start()
    worker.start()
    while worker.running and not enough.is_set():
        time.sleep(0.05)
    worker.stop()
    source.stop()
    if worker.error is not None:
        raise RuntimeError(f"perception failed on {clip}") from worker.error
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CACHE_VERSION, "settings": settings_key(cfg), "clip": str(clip), "frames": frames}
    if max_s is not None:
        payload["max_s"] = max_s
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    return len(frames)


def load_cache(path: Path, cfg: AppConfig | None = None) -> list[CachedFrame]:
    """The cached frames; with `cfg`, refuses a cache built with other perception settings."""
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(f"{path}: cache version {payload.get('version')}, expected {CACHE_VERSION}: rebuild it")
    if cfg is not None and payload.get("settings") != settings_key(cfg):
        raise ValueError(f"{path}: built with other perception settings: rebuild it")
    frames: Sequence[CachedFrame] = payload["frames"]
    return list(frames)


def read_frame(clip: Path, index: int) -> np.ndarray | None:
    """Frame `index` of the clip as BGR (decoded again), or None if it can't be read."""
    cap = cv2.VideoCapture(str(clip))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        return image if ok else None
    finally:
        cap.release()
