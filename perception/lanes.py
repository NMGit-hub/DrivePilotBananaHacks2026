"""YOLOPv2 lane lines and drivable area on the GPU (V2-M5), and the lanes thread.

The model is the official TorchScript release (github.com/CAIC-AD/YOLOPv2, V0.0.1, MIT): one network with three
heads (objects, drivable area, lane lines). Only the two segmentation heads are used here; YOLO26s already
detects objects. Pre- and post-processing follow the official `demo.py` / `utils/utils.py` exactly:

- the frame is resized to 1280x720 (`cv2.resize`, INTER_LINEAR), then letterboxed to 640 wide: resized to 640x360
  and padded with 12 rows of gray (114) above and below, giving the 384x640 input; BGR → RGB, CHW, / 255, FP16;
- the outputs are `[pred, anchor_grid], seg, ll`: `seg` (1, 2, 384, 640) drivable-area scores (argmax over the two
  channels), `ll` (1, 1, 384, 640) lane-line probability (rounded: > 0.5). Both are cropped to rows 12:372 (the
  unpadded image). The official code then upsamples 2x to 1280x720 before the argmax / rounding; here the masks
  stay at the network's own 360x640 (the same decision per pixel, without the interpolation in between), which is
  plenty for lane geometry and 4x cheaper to copy. Mask pixel (row, col) covers frame pixel
  (col * W / 640, row * H / 360): the official code stretches every frame to 16:9 the same way.

On Windows eager PyTorch is bound by kernel-launch overhead (the traced model takes ~20-40 ms a frame, mostly
waiting on launches), so the forward pass is captured once in a CUDA graph and replayed (~13-16 ms on a shared
GPU, identical outputs). The model runs on its own CUDA stream so it overlaps with the perception thread's
detector. Like the detector, it refuses the CPU: the weights must end up on CUDA in the configured precision.

`LaneWorker` runs the model on its own thread on the newest frame only, at most `lanes.fps` times a second (the
pattern of perception/sign_worker.py); the model is built on that thread.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from core.config import LanesConfig
from core.logging_setup import get_logger
from core.types import Frame

log = get_logger(__name__)

# The official pre-processing (demo.py: LoadImages + letterbox): 1280x720, then 640x360 padded to 640x384.
SOURCE_SIZE = (1280, 720)  # (width, height) every frame is resized to first
MASK_SIZE = (640, 360)  # (width, height) of the output masks (the unpadded part of the input)
INPUT_SHAPE = (384, 640)  # network input (height, width)
PAD_TOP = 12  # rows of padding above (and below) the 640x360 image
PAD_VALUE = 114


class GpuRequiredError(RuntimeError):
    """CUDA is unavailable or the lane model did not end up on the GPU as configured."""


@dataclass(frozen=True, slots=True, eq=False)
class LaneMasks:
    """One frame's lane-line and drivable-area masks (360x640 uint8, 1 = yes), in frame coordinates.

    Mask pixel (row, col) covers frame pixel (col * frame_width / 640, row * frame_height / 360).
    """

    t: float
    frame_index: int
    frame_size: tuple[int, int]  # (width, height) of the frame the masks belong to
    lines: np.ndarray  # lane-line pixels
    drivable: np.ndarray  # drivable-area pixels
    timings_ms: dict[str, float]

    @property
    def shape(self) -> tuple[int, int]:
        """Mask (height, width)."""
        return int(self.lines.shape[0]), int(self.lines.shape[1])


def preprocess(image: np.ndarray) -> np.ndarray:
    """The official letterboxed 384x640 BGR uint8 input for one BGR frame (any size; stretched to 16:9)."""
    if (image.shape[1], image.shape[0]) != SOURCE_SIZE:
        image = cv2.resize(image, SOURCE_SIZE, interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(image, MASK_SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        small, PAD_TOP, PAD_TOP, 0, 0, cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE)
    )


def decode(seg: torch.Tensor, ll: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(lane lines, drivable area) as uint8 masks of the unpadded 360x640 image, from the two seg heads."""
    rows = slice(PAD_TOP, PAD_TOP + MASK_SIZE[1])
    drivable = seg[0, :, rows, :].argmax(0).to(torch.uint8)  # official: torch.max over the 2 channels
    lines = (ll[0, 0, rows, :] > 0.5).to(torch.uint8)  # official: torch.round (0.5 rounds to 0)
    return lines, drivable


class LaneModel:
    """YOLOPv2 segmentation on CUDA (FP16 TorchScript, optionally replayed as a CUDA graph).

    Not thread-safe: build and call it on one thread (the lanes thread).
    """

    def __init__(self, cfg: LanesConfig, root: Path) -> None:
        if not torch.cuda.is_available():
            raise GpuRequiredError("CUDA is not available; the lane model runs on the NVIDIA GPU only")
        self.cfg = cfg
        weights = cfg.weights if cfg.weights.is_absolute() else root / cfg.weights
        if not weights.is_file():
            raise FileNotFoundError(f"YOLOPv2 weights not found: {weights} (github.com/CAIC-AD/YOLOPv2 releases)")
        self.device = torch.device(f"cuda:{cfg.device}")
        self.dtype = torch.float16 if cfg.half else torch.float32
        started = time.perf_counter()
        with warnings.catch_warnings():  # torch.jit.load's deprecation notice: the release is TorchScript only
            warnings.simplefilter("ignore", FutureWarning)
            model = torch.jit.load(str(weights), map_location=self.device)
        model = model.half() if cfg.half else model.float()
        self.model: Any = model.eval()
        self._verify_gpu()
        self.stream = torch.cuda.Stream(device=self.device)
        self._input = torch.full((1, 3, *INPUT_SHAPE), PAD_VALUE / 255.0, device=self.device, dtype=self.dtype)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._outputs: Any = None
        self.engine = "torchscript"
        self._warmup()
        log.info(
            "lane model ready",
            weights=weights.name,
            engine=self.engine,
            device=str(self.device),
            precision=str(self.dtype),
            load_s=round(time.perf_counter() - started, 2),
            gpu_mem_mb=round(torch.cuda.memory_allocated(self.device) / 1e6),
        )

    def _verify_gpu(self) -> None:
        """Every parameter on the configured CUDA device in the configured precision, or GpuRequiredError."""
        params = list(self.model.parameters())
        bad = [p for p in params if p.device != self.device or p.dtype != self.dtype]
        if not params or bad:
            seen = {(str(p.device), str(p.dtype)) for p in params}
            raise GpuRequiredError(f"lane model is not on {self.device} in {self.dtype}: {sorted(seen)}")

    def _forward(self) -> Any:
        return self.model(self._input)

    def _warmup(self) -> None:
        """Compile kernels on our stream, then capture the forward pass as a CUDA graph (if configured)."""
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for _ in range(3):
                self._outputs = self._forward()
        self.stream.synchronize()
        if not self.cfg.cuda_graph:
            return
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph, stream=self.stream):
                outputs = self._forward()
            graph.replay()
            self.stream.synchronize()
        except RuntimeError as exc:  # capture can fail on some driver/torch combinations: eager still works
            log.warning("CUDA graph capture failed; running the lane model eagerly", error=str(exc))
            return
        self._graph, self._outputs, self.engine = graph, outputs, "torchscript+cuda_graph"

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """(lane lines, drivable area, stage timings in ms) for one BGR frame. Masks are 360x640 uint8."""
        t0 = time.perf_counter()
        batch = preprocess(image)
        t1 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            pixels = torch.from_numpy(batch).to(self.device, non_blocking=False)
            self._input.copy_(pixels.flip(-1).permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0)
            outputs = self._outputs if self._graph is not None else self._forward()
            if self._graph is not None:
                self._graph.replay()
            (_pred, _anchors), seg, ll = outputs
            lines_t, drivable_t = decode(seg, ll)
            lines = lines_t.cpu().numpy()  # synchronizes the stream
            drivable = drivable_t.cpu().numpy()
        t2 = time.perf_counter()
        return lines, drivable, {"preprocess": (t1 - t0) * 1000, "gpu": (t2 - t1) * 1000, "total": (t2 - t0) * 1000}

    def __call__(self, frame: Frame) -> LaneMasks:
        """Lane masks for one frame."""
        lines, drivable, timings = self.infer(frame.image)
        return LaneMasks(frame.t, frame.index, (frame.width, frame.height), lines, drivable, timings)


class LaneWorker:
    """Runs the lane model on the newest frame, at most `fps` times a second, on its own thread.

    `offer` (any thread, never blocks) hands over a frame; only the newest waits. `on_result(frame, masks)` is
    called on the lanes thread (feed the LaneTracker there). If the model can't be built, the thread ends and
    `error` says why; a failure on one frame is logged and the next frame is tried.
    """

    def __init__(
        self,
        factory: Callable[[], Callable[[Frame], LaneMasks]],
        on_result: Callable[[Frame, LaneMasks], None],
        fps: float,
    ) -> None:
        self._factory = factory
        self._on_result = on_result
        self._interval = 1.0 / fps
        self._cond = threading.Condition()
        self._pending: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ready = threading.Event()  # set once the model is built (or failed: see `error`)
        self.error: str | None = None
        self.processed = 0
        self.last_ms: float | None = None

    def start(self) -> LaneWorker:
        """Start the thread (returns self)."""
        self._thread = threading.Thread(target=self._run, name="lanes", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the thread."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        """The thread is alive (False after a failed start)."""
        return self._thread is not None and self._thread.is_alive()

    def offer(self, frame: Frame) -> None:
        """A new frame (any thread): only the newest waits; never blocks."""
        with self._cond:
            self._pending = frame
            self._cond.notify()

    def _run(self) -> None:
        try:
            model = self._factory()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("lane model could not start")
            self.ready.set()
            return
        self.ready.set()
        next_at = 0.0
        while not self._stop.is_set():
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(0.5)
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            delay = next_at - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
                with self._cond:  # a newer frame arrived while we waited: use that one
                    if self._pending is not None:
                        frame, self._pending = self._pending, None
            next_at = time.perf_counter() + self._interval
            started = time.perf_counter()
            try:
                masks = model(frame)
            except Exception:
                log.exception("lane model failed on a frame", t=frame.t)
                continue
            self.last_ms = (time.perf_counter() - started) * 1000.0
            self.processed += 1
            try:
                self._on_result(frame, masks)
            except Exception:
                log.exception("lane result handling failed", t=frame.t)
