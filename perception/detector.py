"""YOLO26s detection + ByteTrack tracking on the GPU (ultralytics).

The detector runs `YOLO.predict` on CUDA (FP16) and feeds the boxes to ultralytics' `BYTETracker`, which we
drive directly (instead of `YOLO.track`) so the tracker can be reset on seeks and timed separately.

Engine: by default the weights are compiled once into a TensorRT engine (models/<name>_<h>x<w>_fp16.engine).
On Windows, eager PyTorch is bound by kernel-launch overhead (YOLO26s: ~22 ms per frame at 36 % GPU
utilization); TensorRT runs the same network in ~3 ms. Engines are GPU- and TensorRT-version specific, so they
are built locally on first use and never committed. On startup the detector verifies it really runs on the GPU
in the configured precision; it refuses to fall back to the CPU.
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
    # Every raw detection this frame (including low-score and not-yet-confirmed ones), for masking moving objects
    # out of the ego-motion estimate: a crossing truck must not make us look like we're moving.
    raw_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})
HOOD_BOTTOM = 0.97  # a "vehicle" whose box reaches this far down the frame...
HOOD_WIDTH = 0.45  # ...and is this wide is our own hood / dashboard, not another car


def is_own_hood(box: tuple[float, float, float, float], width: int, height: int) -> bool:
    """The detector sometimes takes our own hood or dashboard for a car: wide, at the bottom of the frame."""
    x1, _, x2, y2 = box
    return y2 >= HOOD_BOTTOM * height and (x2 - x1) >= HOOD_WIDTH * width


def vehicle_duplicate_mask(xyxy: np.ndarray, conf: np.ndarray, is_vehicle: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Keep-mask that drops vehicle boxes overlapping a more confident vehicle box (any vehicle class).

    YOLO sometimes reports one pickup as both "car" and "truck"; without this, one vehicle becomes two tracks and
    would be counted twice in the 4-way-stop queue. Non-vehicle boxes are always kept.
    """
    n = len(xyxy)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    # Pairwise IoU in one vectorised step (crowded scenes can have 100+ boxes at the tracker's low threshold).
    x1, y1, x2, y2 = (xyxy[:, k] for k in range(4))
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    iw = (np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :])).clip(0)
    ih = (np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :])).clip(0)
    inter = iw * ih
    union = areas[:, None] + areas[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
    for i in np.argsort(-conf):  # greedy: most confident first
        if not is_vehicle[i] or not keep[i]:
            continue
        suppress = is_vehicle & keep & (iou[i] > iou_thresh)
        suppress[i] = False
        keep[suppress] = False
    return keep


class GpuRequiredError(RuntimeError):
    """CUDA is unavailable or the model did not end up on the GPU."""


class Detector:
    """YOLO26s + ByteTrack. Not thread-safe: call from the perception thread only."""

    def __init__(self, cfg: PerceptionConfig, root: Path, source_fps: float = 30.0) -> None:
        if not torch.cuda.is_available():
            raise GpuRequiredError("CUDA is not available; DrivePilot runs perception on the NVIDIA GPU only")
        self.cfg = cfg
        self._fallback = False  # True when the TensorRT engine is missing and PyTorch weights are used instead
        weights = cfg.weights if cfg.weights.is_absolute() else root / cfg.weights
        weights.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = self._resolve_model(weights)
        self.model = YOLO(str(self.model_path), task="detect")
        names: dict[int, str] = dict(self.model.names)
        by_name = {v: k for k, v in names.items()}
        self.names = names
        self.class_ids = [by_name[c] for c in cfg.classes]
        self.device = f"cuda:{cfg.device}"
        self._predict_args: dict[str, Any] = {
            "device": cfg.device,
            "imgsz": list(cfg.imgsz),
            "conf": min(cfg.conf, cfg.tracker.track_low_thresh),
            "iou": cfg.iou,
            "classes": self.class_ids,
            "verbose": False,
        }
        if cfg.half:
            self._predict_args["quantize"] = 16
        self._source_fps = source_fps
        self._fps_estimate = source_fps  # processed frames per second (EMA), sizes the lost-track window
        self._last_call: float | None = None
        self._id_offset = 0  # ultralytics restarts ids at 1 on reset; offset keeps ids unique across segments
        self._max_id = 0
        self.tracker = self._new_tracker()
        self.warmup()

    def _resolve_model(self, weights: Path) -> Path:
        """The file to load: the TensorRT engine (built by scripts/build_engine.py), or the .pt weights."""
        if self.cfg.engine == "pytorch":
            return weights
        h, w = self.cfg.imgsz
        engine = weights.with_name(f"{weights.stem}_{h}x{w}_{'fp16' if self.cfg.half else 'fp32'}.engine")
        if engine.exists():
            return engine
        # Building needs a separate environment (see scripts/build_engine.py), so it is never done implicitly.
        log.warning(
            "TensorRT engine missing: falling back to PyTorch (~3x slower here). "
            "Build it once with: .venv/Scripts/python scripts/build_engine.py",
            engine=str(engine),
        )
        self._fallback = True
        return weights

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
        """Run a few dummy frames so CUDA kernels are compiled, then verify the weights are on the GPU.

        CUDA state is per thread: the perception thread must call this too (PerceptionWorker does), otherwise
        its first real frame pays a multi-second warm-up.
        """
        dummy = np.zeros((*shape, 3), np.uint8)
        for _ in range(3):
            self.model.predict(dummy, **self._predict_args)
        predictor = self.model.predictor
        assert predictor is not None
        device, precision = self._verify_gpu(predictor.model)
        log.info(
            "detector ready",
            model=self.model_path.name,
            engine="pytorch (fallback)" if self._fallback else self.cfg.engine,
            device=device,
            precision=precision,
            imgsz=list(self.cfg.imgsz),
            classes=list(self.cfg.classes),
            gpu_mem_mb=round(torch.cuda.memory_allocated(self.cfg.device) / 1e6),
        )
        self.reset()

    def _verify_gpu(self, backend: Any) -> tuple[str, str]:
        """Check the loaded network really runs on CUDA in the configured precision. Returns (device, precision)."""
        want_fp16 = self.cfg.half
        if self.cfg.engine == "pytorch" or self._fallback:
            param = next(backend.model.parameters())  # AutoBackend wraps the real network as .model
            ok = param.device.type == "cuda" and param.dtype == (torch.float16 if want_fp16 else torch.float32)
            device, precision = str(param.device), str(param.dtype)
        else:
            impl = backend.backend
            quantize = ((getattr(impl, "metadata", None) or {}).get("args") or {}).get("quantize")
            device = str(backend.device)
            precision = "fp16" if quantize == 16 else "fp32"
            ok = (
                type(impl).__name__ == "TensorRTBackend"
                and backend.device.type == "cuda"
                and (quantize == 16) == want_fp16
            )
        if not ok:
            raise GpuRequiredError(f"detector is not on the GPU as configured: device={device}, precision={precision}")
        return device, precision

    def reset(self) -> None:
        """Forget all tracks (call on a new video segment: seek, loop, source restart). Ids stay unique."""
        self._id_offset = self._max_id
        self._last_call = None
        self.tracker = self._new_tracker()

    def __call__(self, frame: Frame) -> DetectionResult:
        """Detect and track objects in one frame."""
        start = time.perf_counter()
        if self._last_call is not None and start > self._last_call:
            self._fps_estimate = 0.9 * self._fps_estimate + 0.1 / (start - self._last_call)
            # ByteTrack counts its lost-track buffer in processed frames: keep it equal to track_buffer_s seconds.
            self.tracker.max_frames_lost = max(1, round(self.cfg.tracker.track_buffer_s * self._fps_estimate))
        self._last_call = start
        result: Any = self.model.predict(frame.image, **self._predict_args)[0]  # type: ignore[index]
        after_predict = time.perf_counter()
        boxes = result.boxes.cpu().numpy()
        raw_boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes.xyxy]
        if len(boxes):
            is_vehicle = np.array([self.names[int(c)] in VEHICLE_CLASSES for c in boxes.cls])
            keep = vehicle_duplicate_mask(boxes.xyxy, boxes.conf, is_vehicle, self.cfg.vehicle_dedup_iou)
            if not keep.all():
                boxes = boxes[keep]
        tracked = self.tracker.update(boxes, frame.image)
        end = time.perf_counter()

        tracks = [
            Track(
                track_id=int(row[4]) + self._id_offset,
                cls=self.names[int(row[6])],
                conf=float(row[5]),
                box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
            )
            for row in tracked
        ]
        if tracks:
            self._max_id = max(self._max_id, max(t.track_id for t in tracks))
        speed = result.speed  # ultralytics' own per-stage ms (synchronized on CUDA)
        timings = {
            "preprocess": float(speed.get("preprocess") or 0.0),
            "inference": float(speed.get("inference") or 0.0),
            "postprocess": float(speed.get("postprocess") or 0.0),
            "predict_total": (after_predict - start) * 1000,
            "track": (end - after_predict) * 1000,
            "detect_track_total": (end - start) * 1000,
        }
        return DetectionResult(tracks=tracks, timings_ms=timings, raw_boxes=raw_boxes)
