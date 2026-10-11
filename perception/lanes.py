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

Wide cameras. YOLOPv2 is trained on BDD100K (~60° dashcams) and the lane tracker's flat-road geometry assumes a
pinhole view; the phone's 114° ultrawide shows lanes small and bent near the edges. So a camera wider than
`lanes.max_view_deg` gives the lane model only its center, cropped to that field of view (V2_BUILD_PLAN M5's
"center-crop fallback"; no checkerboard calibration has been done), centered across and around `lanes.view_center_y`
down (the user's car dashcam is pitched up: its horizon is ~70 % down the frame, so a crop around the middle would be
mostly sky). `LaneMasks.view` says where the crop sits in the frame; the tracker maps everything it reports back to
frame coordinates.

`LaneWorker` runs the model on its own thread on the newest frame only, at most `lanes.fps` times a second
(perception/frame_worker.py); the model is built on that thread.
"""

from __future__ import annotations

import math
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
from perception.frame_worker import FrameWorker

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
    view: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # the crop the masks cover (frame shares)
    yellow: np.ndarray | None = None  # lane-line pixels painted yellow (None: not known)

    @property
    def shape(self) -> tuple[int, int]:
        """Mask (height, width)."""
        return int(self.lines.shape[0]), int(self.lines.shape[1])


def lane_view(
    hfov_deg: float, max_view_deg: float | None, center_y: float = 0.5
) -> tuple[tuple[float, float, float, float], float]:
    """The crop (x0, y0, x1, y1 as frame shares, same aspect ratio) the lane model sees, and its horizontal field of
    view: the whole frame unless the camera is wider than `max_view_deg` (pinhole model). The crop is centered across
    and around `center_y` down (kept inside the frame): a camera pitched up, with the road low in the frame, wants it
    below the middle."""
    if max_view_deg is None or hfov_deg <= max_view_deg:
        return (0.0, 0.0, 1.0, 1.0), hfov_deg
    share = math.tan(math.radians(max_view_deg) / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    lo, hi = 0.5 - share / 2.0, 0.5 + share / 2.0
    top = min(max(center_y - share / 2.0, 0.0), 1.0 - share)
    return (lo, top, hi, top + share), max_view_deg


def crop_view(image: np.ndarray, view: tuple[float, float, float, float]) -> np.ndarray:
    """The part of the image inside `view` (frame shares)."""
    if view == (0.0, 0.0, 1.0, 1.0):
        return image
    h, w = image.shape[:2]
    x0, y0, x1, y1 = view
    return image[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)]


def preprocess(image: np.ndarray) -> np.ndarray:
    """The official letterboxed 384x640 BGR uint8 input for one BGR frame (any size; stretched to 16:9)."""
    if (image.shape[1], image.shape[0]) != SOURCE_SIZE:
        image = cv2.resize(image, SOURCE_SIZE, interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(image, MASK_SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        small, PAD_TOP, PAD_TOP, 0, 0, cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE)
    )


def yellow_lines(image: np.ndarray, lines: np.ndarray, cfg: LanesConfig) -> np.ndarray:
    """Which lane-line mask pixels are yellow paint in the image (the image the masks were made from, any size):
    hue in `cfg.yellow_hue`, saturation and brightness at least `yellow_min_sat` / `yellow_min_val`. The color mask
    is grown by a pixel (the line mask is a little wider or narrower than the paint). uint8, 1 = yes."""
    small = cv2.resize(image, MASK_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lo = np.array([cfg.yellow_hue[0], cfg.yellow_min_sat, cfg.yellow_min_val], np.uint8)
    hi = np.array([cfg.yellow_hue[1], 255, 255], np.uint8)
    color = cv2.dilate(cv2.inRange(hsv, lo, hi), np.ones((3, 3), np.uint8))
    return ((color > 0) & (lines > 0)).astype(np.uint8)


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

    def __init__(self, cfg: LanesConfig, root: Path, hfov_deg: float | None = None) -> None:
        """`hfov_deg`: the camera's field of view (a wide one is cropped to `cfg.max_view_deg`; None: no crop)."""
        if not torch.cuda.is_available():
            raise GpuRequiredError("CUDA is not available; the lane model runs on the NVIDIA GPU only")
        self.cfg = cfg
        weights = cfg.weights if cfg.weights.is_absolute() else root / cfg.weights
        if not weights.is_file():
            raise FileNotFoundError(f"YOLOPv2 weights not found: {weights} (github.com/CAIC-AD/YOLOPv2 releases)")
        full = (0.0, 0.0, 1.0, 1.0)
        self.view = lane_view(hfov_deg, cfg.max_view_deg, cfg.view_center_y)[0] if hfov_deg is not None else full
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
            # Thread-local: the app's other threads (the detector, the sign model) may use the GPU meanwhile; in
            # the default global mode their calls invalidate the capture (seen in the app: the capture failed and
            # took one of the detector's frames with it).
            with torch.inference_mode(), torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
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
        """Lane masks for one frame (of its center crop, for a wide camera: `view`)."""
        image = crop_view(frame.image, self.view)
        lines, drivable, timings = self.infer(image)
        size = (int(image.shape[1]), int(image.shape[0]))
        yellow = yellow_lines(image, lines, self.cfg)
        return LaneMasks(frame.t, frame.index, size, lines, drivable, timings, self.view, yellow)


class LaneWorker(FrameWorker[LaneMasks]):
    """Runs the lane model on the newest frame, at most `fps` times a second, on its own thread ("lanes").

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
        super().__init__("lanes", factory, on_result, fps)
