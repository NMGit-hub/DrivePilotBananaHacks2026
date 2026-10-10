"""Fine-tuned US traffic-sign detector with a second-stage speed-limit value classifier (V2-M6).

Stage 1: YOLO26s fine-tuned on US signs (13 MUTCD kinds, perception/sign_classes.py) runs on the whole frame at
1280 px wide (imgsz 736x1280), so a distant sign keeps its 15-40 px instead of shrinking to half at 640.
Stage 2: every speed-limit box tall enough to read is cropped from the full-resolution frame and classified into
5-85 mph or "none". A value is reported only when the classifier is confident (`signs.value_conf`) and the box
is at least `signs.value_min_px` tall: a wrong limit is worse than none, so a doubtful value is None.

Both networks run as TensorRT FP16 engines (scripts/signs/export_engines.py) on CUDA; on startup the detector
verifies that, and it refuses to fall back to the CPU. Not thread-safe: call from one thread (the one that
called warmup()).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.config import SignsConfig
from core.logging_setup import get_logger
from core.types import Frame
from perception.sign_classes import KINDS, SPEED_VALUES, VALUE_KINDS, VALUE_REJECT

log = get_logger(__name__)

CROP_PAD = 0.15  # context added around a speed-limit box before classifying (scripts/signs/prepare_data.py)
REFERENCE_WIDTH = 1280  # value_min_px is measured at this frame width


@dataclass(frozen=True, slots=True)
class SignDetection:
    """One detected sign in frame pixel coordinates."""

    kind: str  # one of perception.sign_classes.KINDS: "speed_limit", "school_speed_limit", "stop", ...
    value: int | None  # mph for speed limits when read confidently, else None
    conf: float  # detector confidence
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    value_conf: float | None = None  # classifier probability of `value` (None when no value was attempted)

    def to_record(self) -> dict[str, Any]:
        """Compact JSON-friendly form for run logs."""
        return {
            "kind": self.kind,
            "value": self.value,
            "conf": round(self.conf, 3),
            "value_conf": None if self.value_conf is None else round(self.value_conf, 3),
            "box": [round(v, 1) for v in self.box],
        }


class SignGpuError(RuntimeError):
    """CUDA is unavailable or a sign model did not end up on the GPU in the configured precision."""


def crop_square(image: np.ndarray, box: tuple[float, float, float, float], size: int) -> np.ndarray | None:
    """The classifier's input: the box plus CROP_PAD per side, resized to size x size (None if degenerate).

    Must stay identical to the crops the classifier was trained on (scripts/signs/prepare_data.crop_square).
    """
    x1, y1, x2, y2 = box
    pw, ph = CROP_PAD * (x2 - x1), CROP_PAD * (y2 - y1)
    h, w = image.shape[:2]
    a, b = max(0, round(x1 - pw)), max(0, round(y1 - ph))
    c, d = min(w, round(x2 + pw)), min(h, round(y2 + ph))
    if c - a < 4 or d - b < 4:
        return None
    interp = cv2.INTER_AREA if (c - a) > size else cv2.INTER_LINEAR
    return cv2.resize(image[b:d, a:c], (size, size), interpolation=interp)


def parse_value_class(name: str) -> int | None:
    """Classifier class name -> mph ("v35" -> 35); the reject class and anything unknown -> None."""
    if name == VALUE_REJECT or not name.startswith("v") or not name[1:].isdigit():
        return None
    value = int(name[1:])
    return value if value in SPEED_VALUES else None


def decide_value(probs: np.ndarray, names: dict[int, str], min_prob: float) -> tuple[int | None, float]:
    """(value, probability) from the classifier's probabilities; value None unless top-1 is a speed >= min_prob."""
    top = int(np.argmax(probs))
    p = float(probs[top])
    value = parse_value_class(names[top])
    if value is None or p < min_prob:
        return None, p
    return value, p


def engine_path(weights: Path, imgsz: tuple[int, int], half: bool) -> Path:
    """Where scripts/signs/export_engines.py writes the engine for these weights and input size."""
    return weights.with_name(f"{weights.stem}_{imgsz[0]}x{imgsz[1]}_{'fp16' if half else 'fp32'}.engine")


class SignDetector:
    """US sign detection + speed-value reading on the GPU, one frame at a time."""

    def __init__(self, cfg: SignsConfig, root: Path) -> None:
        if not torch.cuda.is_available():
            raise SignGpuError("CUDA is not available; the sign detector runs on the NVIDIA GPU only")
        unknown = sorted(set(cfg.class_conf) - set(KINDS))
        if unknown:
            raise ValueError(f"signs.class_conf has unknown kinds {unknown}; choose from {list(KINDS)}")
        self.cfg = cfg
        self._fallback: dict[str, bool] = {}
        det_weights = cfg.weights if cfg.weights.is_absolute() else root / cfg.weights
        cls_weights = cfg.classifier_weights if cfg.classifier_weights.is_absolute() else root / cfg.classifier_weights
        s = cfg.classifier_imgsz
        self.det_path = self._resolve("detector", det_weights, cfg.imgsz)
        self.cls_path = self._resolve("classifier", cls_weights, (s, s))
        self.det = YOLO(str(self.det_path), task="detect")
        self.cls = YOLO(str(self.cls_path), task="classify")
        det_names = [self.det.names[i] for i in sorted(self.det.names)]
        if tuple(det_names) != KINDS:
            raise ValueError(f"detector classes {det_names} don't match the sign taxonomy {list(KINDS)}")
        self.cls_names: dict[int, str] = dict(self.cls.names)
        quant: dict[str, Any] = {"quantize": 16} if cfg.half else {}
        self._det_args: dict[str, Any] = {
            "device": cfg.device,
            "imgsz": list(cfg.imgsz),
            "conf": min([cfg.conf, *cfg.class_conf.values()]),
            "iou": cfg.iou,
            "verbose": False,
            **quant,
        }
        self._cls_args: dict[str, Any] = {"device": cfg.device, "imgsz": s, "verbose": False, **quant}
        self.last_timings_ms: dict[str, float] = {}
        self.warmup()

    def _resolve(self, what: str, weights: Path, imgsz: tuple[int, int]) -> Path:
        """The TensorRT engine for these weights, or the weights themselves (pytorch / missing engine)."""
        if self.cfg.engine == "pytorch":
            self._fallback[what] = False
            return weights
        engine = engine_path(weights, imgsz, self.cfg.half)
        if engine.exists():
            self._fallback[what] = False
            return engine
        if not weights.exists():
            raise FileNotFoundError(f"sign {what}: neither {engine.name} nor {weights.name} found in {weights.parent}")
        log.warning(
            f"sign {what} TensorRT engine missing: falling back to PyTorch on the GPU (slower). "
            "Build it with: .venv/Scripts/python scripts/signs/export_engines.py",
            engine=str(engine),
        )
        self._fallback[what] = True
        return weights

    def warmup(self) -> None:
        """Run dummy inputs so CUDA kernels are compiled, then verify both networks run on CUDA as configured.

        CUDA state is per thread: call this from the thread that will call detect().
        """
        dummy = np.zeros((720, 1280, 3), np.uint8)  # a typical dashcam frame
        crop = np.zeros((self.cfg.classifier_imgsz, self.cfg.classifier_imgsz, 3), np.uint8)
        for _ in range(3):
            self.det.predict(dummy, **self._det_args)
            self.cls.predict(crop, **self._cls_args)
        det_dev, det_prec = self._verify_gpu("detector", self.det)
        cls_dev, cls_prec = self._verify_gpu("classifier", self.cls)
        log.info(
            "sign detector ready",
            detector=self.det_path.name,
            classifier=self.cls_path.name,
            device=det_dev,
            precision=det_prec,
            classifier_device=cls_dev,
            classifier_precision=cls_prec,
            imgsz=list(self.cfg.imgsz),
            gpu_mem_mb=round(torch.cuda.memory_allocated(self.cfg.device) / 1e6),
        )

    def _verify_gpu(self, what: str, model: YOLO) -> tuple[str, str]:
        """Check a loaded network really runs on CUDA in the configured precision. Returns (device, precision)."""
        predictor = model.predictor
        assert predictor is not None
        backend: Any = predictor.model
        want_fp16 = self.cfg.half
        if self.cfg.engine == "pytorch" or self._fallback[what]:
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
            raise SignGpuError(f"sign {what} is not on the GPU as configured: device={device}, precision={precision}")
        return device, precision

    def detect(self, frame: Frame) -> list[SignDetection]:
        """Signs in one frame (1280x720 or 1920x1080), most confident first. Timings land in last_timings_ms."""
        t0 = time.perf_counter()
        result: Any = self.det.predict(frame.image, **self._det_args)[0]  # type: ignore[index]
        t1 = time.perf_counter()
        boxes = result.boxes.cpu().numpy()
        scale = REFERENCE_WIDTH / frame.width
        out: list[SignDetection] = []
        n_cls = 0
        for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True):
            kind = KINDS[int(cls)]
            if float(conf) < self.cfg.class_conf.get(kind, self.cfg.conf):
                continue
            box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            value: int | None = None
            value_conf: float | None = None
            if kind in VALUE_KINDS and (box[3] - box[1]) * scale >= self.cfg.value_min_px:
                crop = crop_square(frame.image, box, self.cfg.classifier_imgsz)
                if crop is not None:
                    res: Any = self.cls.predict(crop, **self._cls_args)
                    probs = res[0].probs.data.float().cpu().numpy()
                    value, value_conf = decide_value(probs, self.cls_names, self.cfg.value_conf)
                    n_cls += 1
            out.append(SignDetection(kind, value, float(conf), box, value_conf))
        t2 = time.perf_counter()
        out.sort(key=lambda d: -d.conf)
        speed = result.speed
        self.last_timings_ms = {
            "preprocess": float(speed.get("preprocess") or 0.0),
            "inference": float(speed.get("inference") or 0.0),
            "postprocess": float(speed.get("postprocess") or 0.0),
            "detect_total": (t1 - t0) * 1000,
            "classify_total": (t2 - t1) * 1000,
            "classified": float(n_cls),
            "total": (t2 - t0) * 1000,
        }
        return out
