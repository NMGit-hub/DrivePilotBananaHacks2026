"""Is each tracked vehicle stopped (in the world, not just on screen), and since when?

Per track we keep a short history of two points, the box's bottom-center and top-center, always in the *current*
frame's coordinates: each frame-to-frame homography from ego motion (`EgoMotion.homography`, previous → current)
re-maps the stored points, so our own motion (approaching, creeping forward, the car rocking) is cancelled and
what remains is the vehicle's own motion. All tracks are re-mapped with one `cv2.perspectiveTransform` call.
Homographies are sanity-checked; one within noise of the identity (we are standing still) is skipped so noise
can't accumulate over a long wait. If we move and no homography is available, positions can't be compared any
more: histories and stop anchors are dropped (a car can't look "departed" because *we* moved).

Speed = least-squares slope over `window_s` of the bottom-center position (in box heights, so near and far cars
compare) plus the rate of change of log(box height) (approaching / receding), in box heights per second. Box
jitter from the detector is the main noise, so decisions wait for a full window and the thresholds have a floor
in pixels per second (a 25 px box jittering by 1 px is not "moving").

Derived per track:
- `stopped` / `stopped_since`: hysteresis with a dwell time; the stop is dated from the positions (when it came
  within `still_radius` of where it rests), not from when the noisy speed crossed the threshold.
- `arrival_known`: True only once the car has been *seen moving while we were stopped* (judged by displacement,
  which jitter can't fake). While we drive, the homography cancels our motion only approximately for nearby
  cars, so a car that was already waiting could seem to move and then be dated to our own arrival; instead it
  stays "arrival unknown" and the queue puts it ahead of us (yielding wrongly is the safe mistake).
- `moving_for`: how long it has been moving continuously (departures need sustained motion, not a creep).
- `moved_from_stop`: box heights from where it came to rest (for departures; 0 if that can't be measured).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from core.config import VehicleMotionConfig
from perception.detector import Track

_T, _XB, _YB, _XT, _YT = range(5)  # columns of a history row: time, bottom-center x/y, top-center x/y
_SMOOTH = 9  # samples in the median used for "where is it now" (0.3 s at 30 fps)


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Motion state of one tracked vehicle."""

    track_id: int
    speed: float | None  # box heights per second (None until a full window of history)
    stopped: bool
    stopped_since: float | None
    arrival_known: bool  # False if the car has not moved since it was first seen (it may have been waiting)
    moved_from_stop: float  # box heights moved since it came to rest (0 while there, or if unmeasurable)
    first_seen: float
    seen_moving: bool  # observed moving at least once
    moving_for: float = 0.0  # seconds of continuous motion (0 unless moving)


@dataclass(slots=True)
class _History:
    first_seen: float
    last_seen: float
    rows: np.ndarray  # (n, 5) float64, see the column constants
    stopped: bool = False
    still_since: float | None = None
    moving_since: float | None = None
    stop_anchor: np.ndarray | None = None  # (4,) xb, yb, xt, yt where it came to rest
    origin: np.ndarray | None = None  # (4,) where it was first seen (median of the first samples)
    seen_moving: bool = False
    stationary_from_start: bool = True
    moved_while_watching: bool = False  # measured moving while we stood still


def _height(points: np.ndarray) -> float:
    """Box height from (xb, yb, xt, yt)."""
    xb, yb, xt, yt = points.tolist()
    return max(math.hypot(xb - xt, yb - yt), 1.0)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Box heights between two (xb, yb, xt, yt) positions, plus the log size change."""
    axb, ayb, axt, ayt = a.tolist()
    bxb, byb, bxt, byt = b.tolist()
    h_a, h_b = max(math.hypot(axb - axt, ayb - ayt), 1.0), max(math.hypot(bxb - bxt, byb - byt), 1.0)
    return math.hypot(axb - bxb, ayb - byb) / h_a + abs(math.log(h_a / h_b))


def homography_usable(h_mat: np.ndarray | None, max_shift_px: float, max_scale_change: float) -> bool:
    """A plausible frame-to-frame homography: finite, invertible, and a small motion."""
    if h_mat is None or h_mat.shape != (3, 3):
        return False
    a, b, c, d, e, f, g, k, w = h_mat.ravel().tolist()  # plain floats: numpy is slow on 3x3 checks
    if not all(math.isfinite(v) for v in (a, b, c, d, e, f, g, k, w)) or abs(w) < 1e-9:
        return False
    det = (a * e - b * d) / (w * w)
    if det <= 0 or abs(math.sqrt(det) - 1.0) > max_scale_change:
        return False
    return abs(c / w) <= max_shift_px and abs(f / w) <= max_shift_px and abs(g / w) < 1e-2 and abs(k / w) < 1e-2


def homography_negligible(h_mat: np.ndarray, shift_px: float = 0.25, scale: float = 5e-4) -> bool:
    """Within noise of the identity (standing still): skip it so tiny errors can't add up over a long wait."""
    a, b, c, d, e, f, g, k, _ = (v / h_mat[2, 2] for v in h_mat.ravel().tolist())
    return (
        abs(c) < shift_px
        and abs(f) < shift_px
        and abs(a - 1.0) < scale
        and abs(e - 1.0) < scale
        and abs(b) < scale
        and abs(d) < scale
        and abs(g) < 1e-6
        and abs(k) < 1e-6
    )


def _median_rows(a: np.ndarray) -> np.ndarray:
    """Column-wise median of a small array (np.median's overhead dominates at this size)."""
    s = np.sort(a, axis=0)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) * 0.5


def _slopes(ts: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Least-squares slopes of several series (rows of `series`) against `ts`, in one product.

    Closed form: sum((t - mean t) * y) / sum((t - mean t)^2); the mean of y drops out because the centered
    times sum to zero. ~50x faster than np.polyfit per series for short histories.
    """
    tc = ts - float(np.add.reduce(ts)) / len(ts)
    var = float(tc @ tc)
    return series @ tc / var if var > 0 else np.zeros(len(series))


class VehicleMotionTracker:
    """Per-track motion state. Call `update` once per processed frame; `reset` on a new segment."""

    def __init__(self, cfg: VehicleMotionConfig | None = None) -> None:
        self.cfg = cfg or VehicleMotionConfig()
        self.reset()

    def reset(self) -> None:
        """Forget all tracks."""
        self._tracks: dict[int, _History] = {}
        self._ego_was_stopped = False

    def update(
        self, t: float, tracks: list[Track], homography: np.ndarray | None, ego_stopped: bool
    ) -> dict[int, VehicleState]:
        """Advance with this frame's tracks. Returns the state of every track visible this frame."""
        cfg = self.cfg
        if not homography_usable(homography, cfg.max_shift_px, cfg.max_scale_change):
            homography = None
        if homography is not None and not homography_negligible(homography):
            self._remap_all(homography)
        elif homography is None and not ego_stopped:
            # We moved, but can't say how: stored positions are meaningless now.
            for hist in self._tracks.values():
                hist.rows = hist.rows[:0]
                hist.stop_anchor = None
        if not ego_stopped:
            for hist in self._tracks.values():
                hist.origin = None  # motion evidence for arrival_known only counts while we stand still
        elif not self._ego_was_stopped:
            # We just came to rest: positions from our approach/creep carry compensation error, so cars that are
            # still now get a fresh stop anchor here (they can't look "departed" because of our own motion).
            for hist in self._tracks.values():
                if hist.stopped and len(hist.rows):
                    hist.stop_anchor = self._now(hist).copy()
        self._ego_was_stopped = ego_stopped

        out: dict[int, VehicleState] = {}
        for track in tracks:
            known = self._tracks.get(track.track_id)
            if known is None:
                known = _History(first_seen=t, last_seen=t, rows=np.empty((0, 5)))
                self._tracks[track.track_id] = known
            hist = known
            hist.last_seen = t
            x1, y1, x2, y2 = track.box
            cx = (x1 + x2) / 2.0
            row = np.array([[t, cx, y2, cx, y1]])
            first = int(np.searchsorted(hist.rows[:, _T], t - cfg.window_s))  # rows are in time order
            hist.rows = np.concatenate((hist.rows[first:], row))
            out[track.track_id] = self._advance(track.track_id, hist, t)

        for track_id in [k for k, v in self._tracks.items() if t - v.last_seen > cfg.max_track_age_s]:
            del self._tracks[track_id]
        return out

    # --------------------------------------------------------------------------------- internals

    def _remap_all(self, h_mat: np.ndarray) -> None:
        """Move every stored point (all tracks, anchors, origins) into the current frame: one transform call."""
        blocks: list[np.ndarray] = []
        for hist in self._tracks.values():
            blocks.append(hist.rows[:, 1:5].reshape(-1, 2))
            for extra in (hist.stop_anchor, hist.origin):
                if extra is not None:
                    blocks.append(extra.reshape(-1, 2))
        points = np.concatenate(blocks).astype(np.float64).reshape(-1, 1, 2) if blocks else np.empty((0, 1, 2))
        if len(points) == 0:
            return  # nothing stored yet (OpenCV returns None for an empty input)
        result = cv2.perspectiveTransform(points, h_mat.astype(np.float64))
        if result is None:
            return
        mapped = result.reshape(-1, 2)
        if not np.all(np.isfinite(mapped)):
            return  # degenerate for these points: keep the old coordinates rather than poison them
        i = 0
        for hist in self._tracks.values():
            n = 2 * len(hist.rows)
            hist.rows[:, 1:5] = mapped[i : i + n].reshape(-1, 4)
            i += n
            if hist.stop_anchor is not None:
                hist.stop_anchor = mapped[i : i + 2].reshape(4).copy()
                i += 2
            if hist.origin is not None:
                hist.origin = mapped[i : i + 2].reshape(4).copy()
                i += 2

    def _now(self, hist: _History) -> np.ndarray:
        """Where the vehicle is now: median of the last few samples (robust to box jitter)."""
        return _median_rows(hist.rows[-_SMOOTH:, 1:5])

    def _radius(self, h: float) -> float:
        """Still radius in box heights, with a pixel floor for small boxes."""
        return max(self.cfg.still_radius, self.cfg.still_radius_px / h)

    def _speed(self, hist: _History) -> float | None:
        cfg = self.cfg
        rows = hist.rows
        if len(rows) < cfg.min_samples or rows[-1, _T] - rows[0, _T] < cfg.min_span_frac * cfg.window_s:
            return None
        h = np.maximum(np.hypot(rows[:, _XB] - rows[:, _XT], rows[:, _YB] - rows[:, _YT]), 1.0)
        h_ref = float(np.sort(h)[len(h) // 2])
        series = np.empty((3, len(rows)))
        series[0] = rows[:, _XB] / h_ref
        series[1] = rows[:, _YB] / h_ref
        series[2] = np.log(h)
        vx, vy, vlogh = _slopes(rows[:, _T], series).tolist()
        return math.hypot(vx, vy) + abs(vlogh)

    def _advance(self, track_id: int, hist: _History, t: float) -> VehicleState:
        cfg = self.cfg
        speed = self._speed(hist)
        now = self._now(hist) if len(hist.rows) else None
        h = _height(now) if now is not None else 1.0
        if self._ego_was_stopped and now is not None and hist.origin is None and len(hist.rows) >= _SMOOTH:
            hist.origin = now.copy()  # where it is when we start watching from a standstill

        if speed is not None:
            still_below = max(cfg.stopped_below, cfg.noise_px_per_s / h)
            moving_above = max(cfg.moving_above, cfg.moving_noise_factor * cfg.noise_px_per_s / h)
            if speed < still_below:
                hist.moving_since = None
                if hist.still_since is None:
                    hist.still_since = hist.first_seen if hist.stationary_from_start else self._rest_start(hist, h)
                if not hist.stopped and t - hist.still_since >= cfg.stopped_after_s:
                    hist.stopped = True
                    hist.stop_anchor = self._now(hist).copy()
            elif speed > moving_above:
                hist.seen_moving = True
                if self._ego_was_stopped:
                    hist.moved_while_watching = True
                hist.stopped = False
                hist.still_since = None
                if hist.moving_since is None:
                    hist.moving_since = t
                # stop_anchor is kept: moved_from_stop keeps growing as the car drives off (departures); it is
                # replaced when the car comes to rest somewhere else.
            elif not hist.stopped:
                hist.still_since = None  # in between, while not stopped: not (yet) still

        # "Did it arrive while we watched from a standstill?": it must have both moved away from where we first
        # saw it and been measured moving (jitter can fake one of the two now and then, rarely both).
        if (
            hist.stationary_from_start
            and hist.moved_while_watching
            and hist.origin is not None
            and now is not None
            and _distance(now, hist.origin) > 2.0 * self._radius(h)
        ):
            hist.stationary_from_start = False

        moved = 0.0
        if hist.stop_anchor is not None and now is not None:
            moved = _distance(now, hist.stop_anchor)
        return VehicleState(
            track_id=track_id,
            speed=speed,
            stopped=hist.stopped,
            stopped_since=hist.still_since if hist.stopped else None,
            arrival_known=not hist.stationary_from_start,
            moved_from_stop=moved,
            first_seen=hist.first_seen,
            seen_moving=hist.seen_moving,
            moving_for=0.0 if hist.moving_since is None else t - hist.moving_since,
        )

    def _rest_start(self, hist: _History, h: float) -> float:
        """When the car came to rest: the earliest sample from which it stayed within the still radius of where
        it is now. The fitted speed drops below the threshold only once most of the window is still (up to
        `window_s` late); this dates the stop from the positions instead, which is what arrival order needs."""
        now = self._now(hist)
        radius = self._radius(h)
        start = float(hist.rows[-1, _T])
        for row in hist.rows[::-1]:
            if _distance(row[1:5], now) > radius:
                break
            start = float(row[_T])
        return start
