"""YOLO26s detection + ByteTrack tracking on the GPU (ultralytics).

The detector runs `YOLO.predict` on CUDA (FP16) and feeds the boxes to ultralytics' `BYTETracker`, which we
drive directly (instead of `YOLO.track`) so the tracker can be reset on seeks and timed separately.
On startup it verifies that the network weights really live on the GPU; it refuses to fall back to the CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from core.config import PerceptionConfig
from core.logging_setup import get_logger
from core.types import Frame

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Track:
    """One tracked object in frame pixel coordinates."""

    track_id: int
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def center(self) -> tuple[float, float]:
        """Box center (x, y)."""
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Middle of the bottom edge: where the object touches the road."""
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.box[3] - self.box[1]

    def to_record(self) -> dict[str, Any]:
        """Compact JSON-friendly form for run logs."""
        return {
            "id": self.track_id,
            "cls": self.cls,
            "conf": round(self.conf, 3),
            "box": [round(v, 1) for v in self.box],
        }


@dataclass(slots=True)
class DetectionResult:
    """Tracks for one frame plus stage timings in milliseconds."""

    tracks: list[Track]
    timings_ms: dict[str, float] = field(default_factory=dict)


VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})


def vehicle_duplicate_mask(xyxy: np.ndarray, conf: np.ndarray, is_vehicle: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Keep-mask that drops vehicle boxes overlapping a more confident vehicle box (any vehicle class).

    YOLO sometimes reports one pickup as both "car" and "truck"; without this, one vehicle becomes two tracks and
    would be counted twice in the 4-way-stop queue. Non-vehicle boxes are always kept.
    """
    keep = np.ones(len(xyxy), dtype=bool)
    order = [i for i in np.argsort(-conf) if is_vehicle[i]]
    areas = (xyxy[:, 2] - xyxy[:, 0]).clip(0) * (xyxy[:, 3] - xyxy[:, 1]).clip(0)
    for rank, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[rank + 1 :]:
            if not keep[j]:
                continue
            ix1, iy1 = max(xyxy[i, 0], xyxy[j, 0]), max(xyxy[i, 1], xyxy[j, 1])
            ix2, iy2 = min(xyxy[i, 2], xyxy[j, 2]), min(xyxy[i, 3], xyxy[j, 3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > iou_thresh:
                keep[j] = False
    return keep


class GpuRequiredError(RuntimeError):
    """CUDA is unavailable or the model did not end up on the GPU."""


class Detector:
    """YOLO26s + ByteTrack. Not thread-safe: call from the perception thread only."""

    def __init__(self, cfg: PerceptionConfig, root: Path, source_fps: float = 30.0) -> None:
        if not torch.cuda.is_available():
            raise GpuRequiredError("CUDA is not available; DrivePilot runs perception on the NVIDIA GPU only")
        self.cfg = cfg
        weights = cfg.weights if cfg.weights.is_absolute() else root / cfg.weights
        weights.parent.mkdir(parents=True, exist_ok=True)
        # ultralytics downloads official weights by bare name into the CWD; resolve inside models/ instead.
        self.model = YOLO(str(weights) if weights.exists() else str(weights.parent / weights.name))
        names: dict[int, str] = dict(self.model.names)
        by_name = {v: k for k, v in names.items()}
        self.names = names
        self.class_ids = [by_name[c] for c in cfg.classes]
        self.device = f"cuda:{cfg.device}"
        self._predict_args: dict[str, Any] = {
            "device": cfg.device,
            "imgsz": cfg.imgsz,
            "conf": min(cfg.conf, cfg.tracker.track_low_thresh),
            "iou": cfg.iou,
            "classes": self.class_ids,
            "verbose": False,
        }
        if cfg.half:
            self._predict_args["quantize"] = 16
        self._source_fps = source_fps
        self.tracker = self._new_tracker()
        self.warmup()

    def _new_tracker(self) -> BYTETracker:
        t = self.cfg.tracker
        args = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=t.track_high_thresh,
            track_low_thresh=t.track_low_thresh,
            new_track_thresh=t.new_track_thresh,
            track_buffer=max(1, round(t.track_buffer_s * self._source_fps)),
            match_thresh=t.match_thresh,
            fuse_score=t.fuse_score,
        )
        return BYTETracker(args=args)

    def warmup(self, shape: tuple[int, int] = (720, 1280)) -> None:
        """Run a few dummy frames so CUDA kernels are compiled, then verify the weights are on the GPU."""
        dummy = np.zeros((*shape, 3), np.uint8)
        for _ in range(3):
            self.model.predict(dummy, **self._predict_args)
        predictor = self.model.predictor
        assert predictor is not None
        param = next(predictor.model.model.parameters())  # AutoBackend wraps the real network as .model
        expected_dtype = torch.float16 if self.cfg.half else torch.float32
        if param.device.type != "cuda" or param.dtype != expected_dtype:
            raise GpuRequiredError(f"model weights on {param.device} ({param.dtype}), expected cuda {expected_dtype}")
        log.info(
            "detector ready",
            weights=self.cfg.weights.name,
            device=str(param.device),
            dtype=str(param.dtype),
            imgsz=self.cfg.imgsz,
            classes=list(self.cfg.classes),
            gpu_mem_mb=round(torch.cuda.memory_allocated(self.cfg.device) / 1e6),
        )
        self.reset()

    def reset(self) -> None:
        """Forget all tracks (call on a new video segment: seek, loop, source restart)."""
        self.tracker = self._new_tracker()

    def __call__(self, frame: Frame) -> DetectionResult:
        """Detect and track objects in one frame."""
        start = time.perf_counter()
        result: Any = self.model.predict(frame.image, **self._predict_args)[0]  # type: ignore[index]
        after_predict = time.perf_counter()
        boxes = result.boxes.cpu().numpy()
        if len(boxes):
            is_vehicle = np.array([self.names[int(c)] in VEHICLE_CLASSES for c in boxes.cls])
            keep = vehicle_duplicate_mask(boxes.xyxy, boxes.conf, is_vehicle, self.cfg.vehicle_dedup_iou)
            if not keep.all():
                boxes = boxes[keep]
        tracked = self.tracker.update(boxes, frame.image)
        end = time.perf_counter()

        tracks = [
            Track(
                track_id=int(row[4]),
                cls=self.names[int(row[6])],
                conf=float(row[5]),
                box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
            )
            for row in tracked
        ]
        speed = result.speed  # ultralytics' own per-stage ms (synchronized on CUDA)
        timings = {
            "preprocess": float(speed.get("preprocess") or 0.0),
            "inference": float(speed.get("inference") or 0.0),
            "postprocess": float(speed.get("postprocess") or 0.0),
            "predict_total": (after_predict - start) * 1000,
            "track": (end - after_predict) * 1000,
            "detect_track_total": (end - start) * 1000,
        }
        return DetectionResult(tracks=tracks, timings_ms=timings)
