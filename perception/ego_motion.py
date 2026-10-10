"""Are we stopped? Ego motion from sparse optical flow on a downscaled grayscale frame.

Features are tracked with pyramidal Lucas-Kanade inside a region of interest that excludes the sky (few
features) and the hood/dashboard (moves with the camera, so it always looks still). Detected objects are masked
out because they move on their own; a car crossing in front of us must not make us look "moving". Tracks are
kept only if the backward flow returns to the start point (forward-backward check).

Motion is measured over two baselines:
- short: between consecutive frames (fast to react; also gives the frame-to-frame homography);
- long: from the frame `baseline_s` ago straight to the current frame. Real motion accumulates over the baseline
  while tracking noise does not, so the noise floor while stopped drops ~10x (rain shimmer, compression noise),
  which is what lets slow crawling be told apart from standing still.
The state machine uses the long measure, falling back to the short one when the long one can't be measured
(e.g. the scene moved too far to track while pulling away quickly).

Measures (per second, so independent of the frame rate):
- `zoom`: |log(scale)| of a RANSAC similarity transform (rotation + uniform scale + shift). Driving forward
  makes the scene expand; pitch rocking after braking and steering only shift/rotate it.
- `horizontal`: median |dx| of the tracked points (px/s at `work_width`).
- `magnitude`: median |d| of the tracked points (px/s at `work_width`).
- `combined` (default): max(zoom × zoom_weight, horizontal). Zoom alone misses slow turns (the car yaws, the
  scene slides sideways but barely expands); horizontal flow alone is disturbed by rocking on hills.
Calibrated against hand-labeled stops: scripts/eval_ego_motion.py with eval/ego_stop_labels.yaml.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from core.config import EgoMotionConfig
from core.types import Frame

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class MotionMeasures:
    """Raw motion measures over one baseline (None when not measurable)."""

    zoom: float | None  # |log scale| per second
    horizontal: float | None  # px/s at work_width
    magnitude: float | None  # px/s at work_width
    n_points: int
    baseline_s: float  # actual time between the two frames compared (0 = not measured)

    def get(self, metric: str, zoom_weight: float = 50.0) -> float | None:
        """The measure named `metric` ('combined' = max(zoom × zoom_weight, horizontal))."""
        if metric == "combined":
            if self.zoom is None or self.horizontal is None:
                return None
            return max(self.zoom * zoom_weight, self.horizontal)
        value: float | None = getattr(self, metric)
        return value


NOT_MEASURED = MotionMeasures(None, None, None, 0, 0.0)


@dataclass(frozen=True, slots=True)
class EgoMotion:
    """Ego-motion estimate for one frame."""

    t: float
    short: MotionMeasures  # consecutive frames
    long: MotionMeasures  # current frame vs the frame `baseline_s` ago
    value: float | None  # smoothed measure that drove the state (None = not measurable this frame)
    stopped: bool
    stopped_since: float | None  # media time the current stillness began (None while moving)
    confident: bool  # False when nothing was measurable and the state was held
    homography: np.ndarray | None  # 3x3 prev→curr in full-resolution pixels, None if unavailable
    elapsed_ms: float


class StopStateMachine:
    """Hysteresis on a smoothed motion measure.

    moving --(smoothed value < stopped_below continuously for stopped_after_s)--> stopped
    stopped --(smoothed value > moving_above)--> moving
    `stopped_since` is when the stillness began, not when it was confirmed: that is our arrival time at a stop.
    Values are smoothed with a rolling median over `smoothing_s`, so a single glitchy frame can't flip the state.
    """

    def __init__(self, stopped_below: float, moving_above: float, stopped_after_s: float, smoothing_s: float) -> None:
        self.stopped_below = stopped_below
        self.moving_above = moving_above
        self.stopped_after_s = stopped_after_s
        self.smoothing_s = smoothing_s
        self.reset()

    def reset(self) -> None:
        """Forget history (new segment)."""
        self._window: deque[tuple[float, float]] = deque()
        self.stopped = False
        self._still_since: float | None = None

    @property
    def stopped_since(self) -> float | None:
        """Start of the current stop, or None while moving."""
        return self._still_since if self.stopped else None

    def update(self, t: float, value: float | None) -> float | None:
        """Feed one measurement (None = unmeasurable: state held). Returns the smoothed value used."""
        if value is None:
            return None
        self._window.append((t, value))
        while self._window and t - self._window[0][0] > self.smoothing_s:
            self._window.popleft()
        smoothed = float(np.median([v for _, v in self._window]))
        if smoothed < self.stopped_below:
            if self._still_since is None:
                # Stillness began roughly when the first still sample of the smoothing window was taken.
                self._still_since = next((ts for ts, v in self._window if v < self.stopped_below), t)
            if not self.stopped and t - self._still_since >= self.stopped_after_s:
                self.stopped = True
        elif smoothed > self.moving_above:
            self.stopped = False
            self._still_since = None
        elif not self.stopped:
            self._still_since = None  # in the hysteresis band while moving: not (yet) still
        return smoothed


@dataclass(slots=True)
class _Keyframe:
    t: float
    gray: np.ndarray
    points: np.ndarray | None  # features detected in this frame (inside its mask)


class EgoMotionEstimator:
    """Stateful per-stream estimator; call `update` once per processed frame (any frame rate)."""

    def __init__(self, cfg: EgoMotionConfig) -> None:
        self.cfg = cfg
        self._lk_win = (21, 21)
        self._lk_levels = 4  # long baselines need large displacements
        self._lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self.state = StopStateMachine(cfg.stopped_below, cfg.moving_above, cfg.stopped_after_s, cfg.smoothing_s)
        self.reset()

    def reset(self) -> None:
        """Forget history (new video segment)."""
        self._history: deque[_Keyframe] = deque()
        self._scale = 1.0
        self.state.reset()

    def _prepare(self, frame: Frame, boxes: Sequence[Box]) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        h, w = frame.image.shape[:2]
        self._scale = cfg.work_width / w
        small_h = max(1, round(h * self._scale))
        gray = cv2.cvtColor(
            cv2.resize(frame.image, (cfg.work_width, small_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
        )
        mask = np.zeros_like(gray)
        mask[int(cfg.roi_top * small_h) : int(cfg.roi_bottom * small_h), :] = 255
        for x1, y1, x2, y2 in boxes:
            mx, my = (x2 - x1) * cfg.box_margin, (y2 - y1) * cfg.box_margin
            sx1, sy1 = max(0, int((x1 - mx) * self._scale)), max(0, int((y1 - my) * self._scale))
            sx2, sy2 = int(np.ceil((x2 + mx) * self._scale)), int(np.ceil((y2 + my) * self._scale))
            mask[sy1:sy2, sx1:sx2] = 0
        return gray, mask

    def measure(
        self, frame: Frame, boxes: Sequence[Box] = ()
    ) -> tuple[MotionMeasures, MotionMeasures, np.ndarray | None]:
        """Short and long measures plus the frame-to-frame homography (advances the frame history)."""
        gray, mask = self._prepare(frame, boxes)
        if self._history and self._history[-1].gray.shape != gray.shape:
            self.reset()  # the frame size changed (new camera session): old features are in other coordinates
        short, long, homography = NOT_MEASURED, NOT_MEASURED, None
        if self._history and frame.t > self._history[-1].t:
            prev = self._history[-1]
            short, p0, p1 = self._measure_between(prev, gray, mask, frame.t)
            if p0 is not None and p1 is not None:
                homography = self._homography(p0, p1)
            base = self._baseline_keyframe(frame.t)
            if base is not None:
                long, _, _ = self._measure_between(base, gray, mask, frame.t)

        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.cfg.max_points, qualityLevel=0.01, minDistance=7, mask=mask, blockSize=7
        )
        self._history.append(_Keyframe(frame.t, gray, points))
        while len(self._history) > 2 and frame.t - self._history[1].t >= self.cfg.baseline_s:
            self._history.popleft()  # keep exactly one keyframe at least baseline_s old
        return short, long, homography

    def _baseline_keyframe(self, t: float) -> _Keyframe | None:
        """The most recent stored frame at least `baseline_s` old (None until enough history)."""
        best = None
        for keyframe in self._history:
            if t - keyframe.t >= self.cfg.baseline_s - 1e-6:
                best = keyframe
        return best

    def _measure_between(
        self, ref: _Keyframe, gray: np.ndarray, mask: np.ndarray, t: float
    ) -> tuple[MotionMeasures, np.ndarray | None, np.ndarray | None]:
        if ref.points is None or len(ref.points) < self.cfg.min_points:
            return NOT_MEASURED, None, None
        p0, p1 = self._track(ref.gray, gray, ref.points, mask)
        if len(p0) < self.cfg.min_points:
            return NOT_MEASURED, None, None
        # Guard against near-duplicate timestamps (bursty live frames): rates need a sane time step.
        dt = max(t - ref.t, self.cfg.min_dt_s)
        d = p1 - p0
        measures = MotionMeasures(
            zoom=self._zoom_rate(p0, p1, dt),
            horizontal=float(np.median(np.abs(d[:, 0]))) / dt,
            magnitude=float(np.median(np.linalg.norm(d, axis=1))) / dt,
            n_points=len(p0),
            baseline_s=t - ref.t,
        )
        return measures, p0, p1

    def update(self, frame: Frame, boxes: Sequence[Box] = ()) -> EgoMotion:
        """Process one frame. `boxes` are detected objects (x1, y1, x2, y2) in full-resolution pixels."""
        start = time.perf_counter()
        short, long, homography = self.measure(frame, boxes)
        raw = long.get(self.cfg.metric, self.cfg.zoom_weight)
        if raw is None:  # long baseline not available yet, or the scene moved too far to track
            raw = short.get(self.cfg.metric, self.cfg.zoom_weight)
        smoothed = self.state.update(frame.t, raw)
        return EgoMotion(
            t=frame.t,
            short=short,
            long=long,
            value=smoothed,
            stopped=self.state.stopped,
            stopped_since=self.state.stopped_since,
            confident=raw is not None,
            homography=homography,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

    def _track(
        self, prev: np.ndarray, curr: np.ndarray, prev_pts: np.ndarray, curr_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """LK forward + backward; keep points that round-trip and land outside masked objects."""
        lk = {"winSize": self._lk_win, "maxLevel": self._lk_levels, "criteria": self._lk_criteria}
        curr_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev, curr, prev_pts, None, **lk)  # type: ignore[call-overload]
        back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(curr, prev, curr_pts, None, **lk)  # type: ignore[call-overload]
        fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).ravel()
        ok = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_error < self.cfg.fb_max_error_px)
        p0, p1 = prev_pts.reshape(-1, 2)[ok], curr_pts.reshape(-1, 2)[ok]
        h, w = curr_mask.shape
        xi = np.clip(p1[:, 0].round().astype(int), 0, w - 1)
        yi = np.clip(p1[:, 1].round().astype(int), 0, h - 1)
        inside = curr_mask[yi, xi] > 0  # drop points that moved onto an object in the current frame
        return p0[inside], p1[inside]

    @staticmethod
    def _zoom_rate(p0: np.ndarray, p1: np.ndarray, dt: float) -> float | None:
        matrix, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=0.5)
        if matrix is None:
            return None
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        return abs(math.log(scale)) / dt if scale > 0 else None

    def _homography(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray | None:
        if len(p0) < 12:
            return None
        h_small, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 1.0)
        if h_small is None:
            return None
        s = np.diag([self._scale, self._scale, 1.0])
        return np.linalg.inv(s) @ h_small @ s  # full-resolution coordinates
