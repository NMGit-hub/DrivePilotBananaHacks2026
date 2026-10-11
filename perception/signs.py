"""Fine-tuned US traffic-sign detector with a second-stage speed-limit value classifier (V2-M6).

Stage 1: YOLO26s fine-tuned on US signs (13 MUTCD kinds, perception/sign_classes.py) runs on the whole frame at
1280 px wide (imgsz 736x1280), so a distant sign keeps its 15-40 px instead of shrinking to half at 640.
Stage 2: every speed-limit box tall enough to read is cropped from the full-resolution frame and classified into
5-85 mph or "none". A value is reported only when the classifier is confident (`signs.value_conf`), the box is at
least `signs.value_min_px` tall and doesn't touch the frame border: a wrong limit is worse than none, so a doubtful
value is None.

`SignDetector.raw_detections()` returns every box with the classifier's top reading; `select()` applies the app's
rules (per-kind confidence, duplicates, the value rule). `detect()` is the two together, and the evaluation
(scripts/signs/evaluate.py) calls the same `select()`, so measured numbers are the app's numbers.

Both networks run as TensorRT FP16 engines (scripts/signs/export_engines.py) on CUDA; on startup the detector
verifies that, and it refuses to fall back to the CPU. An engine built from other weights than the configured ones
(its sidecar JSON records their sha256) is not used. Not thread-safe: call from one thread (the one that called
warmup()).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

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
REFERENCE_WIDTH = 1280  # value_min_px and value_edge_px are measured at this frame width
Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SignDetection:
    """One detected sign in frame pixel coordinates."""

    kind: str  # one of perception.sign_classes.KINDS: "speed_limit", "school_speed_limit", "stop", ...
    value: int | None  # mph for speed limits when read confidently, else None
    conf: float  # detector confidence
    box: Box  # x1, y1, x2, y2 in frame pixels
    value_conf: float | None = None  # classifier's top-class probability (None when the value wasn't attempted)

    def to_record(self) -> dict[str, Any]:
        """Compact JSON-friendly form for run logs."""
        return {
            "kind": self.kind,
            "value": self.value,
            "conf": round(self.conf, 3),
            "value_conf": None if self.value_conf is None else round(self.value_conf, 3),
            "box": [round(v, 1) for v in self.box],
        }


@dataclass(frozen=True, slots=True)
class RawSign:
    """One detector box before the app's rules, with the classifier's top reading if it was classified.

    top_value is the top-1 class as mph (None when the top class is the reject class "none"); top_prob is its
    probability. Both are None when the box wasn't classified (not a speed kind, too small, degenerate crop).
    """

    kind: str
    conf: float
    box: Box
    top_value: int | None = None
    top_prob: float | None = None


class SignGpuError(RuntimeError):
    """CUDA is unavailable or a sign model did not end up on the GPU in the configured precision."""


def crop_square(image: np.ndarray, box: Box, size: int) -> np.ndarray | None:
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


VALUE_CLASS_NAMES: frozenset[str] = frozenset({*(f"v{v}" for v in SPEED_VALUES), VALUE_REJECT})


def check_value_classes(names: dict[int, str]) -> None:
    """Raise ValueError unless the classifier's classes are exactly v5 ... v85 plus the reject class."""
    got = set(names.values())
    if got != VALUE_CLASS_NAMES or len(names) != len(VALUE_CLASS_NAMES):
        missing, extra = sorted(VALUE_CLASS_NAMES - got), sorted(got - VALUE_CLASS_NAMES)
        raise ValueError(
            f"value classifier classes don't match v5..v85 + {VALUE_REJECT!r}: missing {missing}, extra {extra}"
        )


def top_reading(probs: np.ndarray, names: dict[int, str]) -> tuple[int | None, float]:
    """(top-1 value or None for the reject class, its probability) from the classifier's probabilities."""
    top = int(np.argmax(probs))
    return parse_value_class(names[top]), float(probs[top])


def decide_value(probs: np.ndarray, names: dict[int, str], min_prob: float) -> tuple[int | None, float]:
    """(value, probability) from the classifier's probabilities; value None unless top-1 is a speed >= min_prob."""
    value, p = top_reading(probs, names)
    if value is None or p < min_prob:
        return None, p
    return value, p


def _iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class _Scored(Protocol):
    @property
    def conf(self) -> float: ...

    @property
    def box(self) -> Box: ...


_S = TypeVar("_S", bound=_Scored)


def drop_duplicates(dets: Sequence[_S], iou: float) -> list[_S]:
    """Most confident first, without boxes that overlap a more confident detection of any kind by more than iou."""
    kept: list[_S] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(_iou(d.box, k.box) <= iou for k in kept):
            kept.append(d)
    return kept


def value_attempted(kind: str, box: Box, frame_w: int, min_px: float) -> bool:
    """Whether the app tries to read a value: a speed kind at least min_px tall (measured at 1280 px width)."""
    return kind in VALUE_KINDS and (box[3] - box[1]) * REFERENCE_WIDTH / frame_w >= min_px


def touches_border(box: Box, frame_w: int, frame_h: int, edge_px: float) -> bool:
    """Whether a box lies within edge_px (measured at 1280 px width) of the frame border: its number may be cut."""
    e = edge_px * frame_w / REFERENCE_WIDTH
    return box[0] < e or box[1] < e or box[2] > frame_w - e or box[3] > frame_h - e


def select(raw: Sequence[RawSign], cfg: SignsConfig, frame_w: int, frame_h: int) -> list[SignDetection]:
    """The app's rules on raw detections: per-kind confidence, then duplicates, then the value rule.

    A value is reported only for a speed kind at least value_min_px tall, not within value_edge_px of the border,
    whose classifier top-1 is a speed with probability >= value_conf. Most confident first.
    """
    passing = [r for r in raw if r.conf >= cfg.class_conf.get(r.kind, cfg.conf)]
    out: list[SignDetection] = []
    for r in drop_duplicates(passing, cfg.duplicate_iou):
        value: int | None = None
        value_conf: float | None = None
        if r.top_prob is not None and value_attempted(r.kind, r.box, frame_w, cfg.value_min_px):
            value_conf = r.top_prob
            if (
                r.top_value is not None
                and r.top_prob >= cfg.value_conf
                and not touches_border(r.box, frame_w, frame_h, cfg.value_edge_px)
            ):
                value = r.top_value
        out.append(SignDetection(r.kind, value, r.conf, r.box, value_conf))
    return out


def engine_path(weights: Path, imgsz: tuple[int, int], half: bool) -> Path:
    """Where scripts/signs/export_engines.py writes the engine for these weights and input size."""
    return weights.with_name(f"{weights.stem}_{imgsz[0]}x{imgsz[1]}_{'fp16' if half else 'fp32'}.engine")


def sidecar_path(engine: Path) -> Path:
    """The JSON next to an engine recording what it was built from (written by scripts/signs/export_engines.py)."""
    return engine.with_name(engine.name + ".json")


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_staleness(engine: Path, weights: Path) -> str | None:
    """Why an existing engine must not be used for these weights, or None if it may.

    Stale when its sidecar records other weights (sha256), or, without a sidecar, when the weights are newer than
    the engine. Without the weights file there is nothing to compare: the engine is used.
    """
    if not weights.exists():
        return None
    side = sidecar_path(engine)
    if side.exists():
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return f"unreadable sidecar {side.name}: {e}"
        recorded = data.get("weights_sha256") if isinstance(data, dict) else None
        if recorded != file_sha256(weights):
            return f"built from other weights than {weights.name} (sha256 differs)"
        return None
    if weights.stat().st_mtime > engine.stat().st_mtime:
        return f"{weights.name} is newer than the engine and no sidecar records what it was built from"
    return None


def engine_layer_names(engine: Any) -> list[str]:
    """Layer names of a deserialized TensorRT engine (empty if the inspector gives none)."""
    import tensorrt as trt

    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", []) if isinstance(info, dict) else []
    return [layer if isinstance(layer, str) else str(layer.get("Name", "")) for layer in layers]


def engine_is_fp16(io_fp16: bool, metadata_quantize: Any, layer_names: Sequence[str]) -> bool:
    """Whether a TensorRT engine computes in FP16.

    Exported with quantize=16 (ultralytics metadata) and either FP16 input bindings or, for ModelOpt AutoCast
    engines that keep FP32 I/O, a cast of the input to FP16 inside the network ("images_cast_to_fp16").
    """
    return metadata_quantize == 16 and (io_fp16 or any("fp16" in n.lower() for n in layer_names))


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
        check_value_classes(self.cls_names)
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
        """The TensorRT engine for these weights, or the weights themselves (pytorch / missing or stale engine)."""
        if self.cfg.engine == "pytorch":
            self._fallback[what] = False
            return weights
        engine = engine_path(weights, imgsz, self.cfg.half)
        reason = "missing"
        if engine.exists():
            stale = engine_staleness(engine, weights)
            if stale is None:
                self._fallback[what] = False
                return engine
            reason = f"stale ({stale})"
        if not weights.exists():
            raise FileNotFoundError(f"sign {what}: neither {engine.name} nor {weights.name} found in {weights.parent}")
        log.warning(
            f"sign {what} TensorRT engine {reason}: falling back to PyTorch on the GPU (slower). "
            "Build it with: .venv/Scripts/python scripts/signs/export_engines.py --force",
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
        free, total = torch.cuda.mem_get_info(self.cfg.device)  # whole device: torch's counters miss TensorRT
        log.info(
            "sign detector ready",
            detector=self.det_path.name,
            classifier=self.cls_path.name,
            device=det_dev,
            precision=det_prec,
            classifier_device=cls_dev,
            classifier_precision=cls_prec,
            imgsz=list(self.cfg.imgsz),
            gpu_used_mb=round((total - free) / 1e6),
            gpu_total_mb=round(total / 1e6),
        )

    def _verify_gpu(self, what: str, model: YOLO) -> tuple[str, str]:
        """Check a loaded network really runs on CUDA in the configured precision. Returns (device, precision)."""
        predictor = model.predictor
        if predictor is None:
            raise SignGpuError(f"sign {what} has no predictor yet: run warmup() first")
        backend: Any = predictor.model
        want_fp16 = self.cfg.half
        if self.cfg.engine == "pytorch" or self._fallback[what]:
            param = next(backend.model.parameters())  # AutoBackend wraps the real network as .model
            ok = param.device.type == "cuda" and param.dtype == (torch.float16 if want_fp16 else torch.float32)
            device, precision = str(param.device), str(param.dtype)
        else:
            impl = backend.backend
            is_trt = type(impl).__name__ == "TensorRTBackend"
            quantize = ((getattr(impl, "metadata", None) or {}).get("args") or {}).get("quantize")
            layers = engine_layer_names(impl.model) if is_trt else []
            fp16 = is_trt and engine_is_fp16(bool(getattr(impl, "fp16", False)), quantize, layers)
            device = str(backend.device)
            precision = "fp16" if fp16 else "fp32"
            ok = is_trt and backend.device.type == "cuda" and fp16 == want_fp16
        if not ok:
            raise SignGpuError(f"sign {what} is not on the GPU as configured: device={device}, precision={precision}")
        return device, precision

    def raw_detections(self, frame: Frame, classify_min_px: float) -> list[RawSign]:
        """Every box at the detector's lowest configured confidence, speed kinds classified if tall enough.

        classify_min_px is measured at 1280 px frame width (the app passes value_min_px; the evaluation passes 1
        so it can sweep the rules offline). Timings land in last_timings_ms.
        """
        t0 = time.perf_counter()
        result: Any = self.det.predict(frame.image, **self._det_args)[0]  # type: ignore[index]
        t1 = time.perf_counter()
        boxes = result.boxes.cpu().numpy()
        out: list[RawSign] = []
        n_cls = 0
        for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True):
            kind = KINDS[int(cls)]
            box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            top_value: int | None = None
            top_prob: float | None = None
            if value_attempted(kind, box, frame.width, classify_min_px):
                crop = crop_square(frame.image, box, self.cfg.classifier_imgsz)
                if crop is not None:
                    res: Any = self.cls.predict(crop, **self._cls_args)
                    top_value, top_prob = top_reading(res[0].probs.data.float().cpu().numpy(), self.cls_names)
                    n_cls += 1
            out.append(RawSign(kind, float(conf), box, top_value, top_prob))
        t2 = time.perf_counter()
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

    def detect(self, frame: Frame) -> list[SignDetection]:
        """Signs in one frame (1280x720 or 1920x1080), most confident first. Timings land in last_timings_ms."""
        t0 = time.perf_counter()
        raw = self.raw_detections(frame, self.cfg.value_min_px)
        out = select(raw, self.cfg, frame.width, frame.height)
        self.last_timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return out
