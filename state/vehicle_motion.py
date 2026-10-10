"""Is each tracked vehicle stopped (in the world, not just on screen), and since when?

For every track we keep a short history of its box, always expressed in the *current* frame's coordinates:
each new frame-to-frame homography from ego motion (`EgoMotion.homography`) re-maps the stored points, so our
own motion (approaching the intersection) is approximately cancelled and what remains is the car's own motion.
The homography is only exact for distant/planar scenery, so while we move the estimate is coarser; once we
are stopped it is exact (H = identity).

Speed is the least-squares slope over `window_s` of the box's bottom-center position (normalized by the box
height, so it is the same for near and far cars) plus the rate of change of log(box height) (approaching or
receding). Units: box heights per second. A hysteresis state machine turns speed into stopped/moving with a
dwell time; `stopped_since` is when the stillness began. `moved_from_stop` measures how far a car has moved
since it stopped (for departures).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from core.config import VehicleMotionConfig
from perception.detector import Track


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Motion state of one tracked vehicle."""

    track_id: int
    speed: float | None  # box heights per second (None until enough history)
    stopped: bool
    stopped_since: float | None
    arrival_known: bool  # False if the car was already stationary when first seen (it may have been waiting)
    moved_from_stop: float  # box heights moved since it stopped (0 while stopped at the same spot)
    first_seen: float
    seen_moving: bool  # observed moving at least once


@dataclass(slots=True)
class _History:
    first_seen: float
    last_seen: float
    samples: deque[tuple[float, float, float, float]] = field(default_factory=deque)  # t, x_bc, y_bc, h
    stopped: bool = False
    still_since: float | None = None
    stop_anchor: tuple[float, float, float] | None = None  # x_bc, y_bc, h where it came to rest
    seen_moving: bool = False
    stationary_from_start: bool = True


def _remap(h_mat: np.ndarray, x: float, y: float) -> tuple[float, float, float]:
    """Map a point through a homography; also return the local scale factor (for box sizes)."""
    p = h_mat @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return x, y, 1.0
    nx, ny = p[0] / p[2], p[1] / p[2]
    # Local scale from the Jacobian determinant at (x, y).
    eps = 1.0
    px = h_mat @ np.array([x + eps, y, 1.0])
    py = h_mat @ np.array([x, y + eps, 1.0])
    if abs(px[2]) < 1e-9 or abs(py[2]) < 1e-9:
        return nx, ny, 1.0
    jx = np.array([px[0] / px[2] - nx, px[1] / px[2] - ny])
    jy = np.array([py[0] / py[2] - nx, py[1] / py[2] - ny])
    det = abs(jx[0] * jy[1] - jx[1] * jy[0])
    return float(nx), float(ny), math.sqrt(det) if det > 0 else 1.0


class VehicleMotionTracker:
    """Per-track motion state. Call `update` once per processed frame; `reset` on a new segment."""

    def __init__(self, cfg: VehicleMotionConfig | None = None) -> None:
        self.cfg = cfg or VehicleMotionConfig()
        self.reset()

    def reset(self) -> None:
        """Forget all tracks."""
        self._tracks: dict[int, _History] = {}

    def update(
        self, t: float, tracks: list[Track], homography: np.ndarray | None, ego_stopped: bool
    ) -> dict[int, VehicleState]:
        """Advance with this frame's tracks. Returns the state of every track visible this frame."""
        cfg = self.cfg
        compensate = homography is not None and not ego_stopped
        for hist in self._tracks.values():
            if compensate:
                assert homography is not None
                hist.samples = deque((ts, *self._remapped(homography, x, y, hh)) for ts, x, y, hh in hist.samples)
                if hist.stop_anchor is not None:
                    ax, ay, ah = hist.stop_anchor
                    nx, ny, scale = _remap(homography, ax, ay)
                    hist.stop_anchor = (nx, ny, ah * scale)
            elif homography is None and not ego_stopped:
                hist.samples.clear()  # we moved but can't say how: old positions are meaningless now

        out: dict[int, VehicleState] = {}
        for track in tracks:
            x_bc, y_bc = track.bottom_center
            h = max(track.height, 1.0)
            existing = self._tracks.get(track.track_id)
            hist = existing if existing is not None else _History(first_seen=t, last_seen=t)
            self._tracks[track.track_id] = hist
            hist.last_seen = t
            hist.samples.append((t, x_bc, y_bc, h))
            while hist.samples and t - hist.samples[0][0] > cfg.window_s:
                hist.samples.popleft()
            speed = self._speed(hist)
            self._advance(hist, t, speed)
            moved = 0.0
            if hist.stop_anchor is not None:
                ax, ay, ah = hist.stop_anchor
                moved = math.hypot(x_bc - ax, y_bc - ay) / h + abs(math.log(h / max(ah, 1.0)))
            out[track.track_id] = VehicleState(
                track_id=track.track_id,
                speed=speed,
                stopped=hist.stopped,
                stopped_since=hist.still_since if hist.stopped else None,
                arrival_known=not hist.stationary_from_start,
                moved_from_stop=moved,
                first_seen=hist.first_seen,
                seen_moving=hist.seen_moving,
            )

        for track_id in [k for k, v in self._tracks.items() if t - v.last_seen > cfg.max_track_age_s]:
            del self._tracks[track_id]
        return out

    @staticmethod
    def _remapped(h_mat: np.ndarray, x: float, y: float, h: float) -> tuple[float, float, float]:
        nx, ny, scale = _remap(h_mat, x, y)
        return nx, ny, h * scale

    def _speed(self, hist: _History) -> float | None:
        if len(hist.samples) < self.cfg.min_samples:
            return None
        arr = np.asarray(hist.samples)
        ts = arr[:, 0] - arr[0, 0]
        if ts[-1] <= 0:
            return None
        h_ref = float(np.median(arr[:, 3]))
        vx = np.polyfit(ts, arr[:, 1] / h_ref, 1)[0]
        vy = np.polyfit(ts, arr[:, 2] / h_ref, 1)[0]
        vlogh = np.polyfit(ts, np.log(arr[:, 3]), 1)[0]
        return float(math.hypot(vx, vy) + abs(vlogh))

    def _rest_start(self, hist: _History) -> float:
        """When the car came to rest: the earliest sample from which it stayed within `still_radius` of where it
        is now. The fitted speed drops below the threshold only once the window is mostly still (up to
        `window_s` late); this dates the stop from the positions instead, which is what arrival order needs."""
        _, x0, y0, h0 = hist.samples[-1]
        start = hist.samples[-1][0]
        for ts, x, y, h in reversed(hist.samples):
            if math.hypot(x - x0, y - y0) / h0 + abs(math.log(h / h0)) > self.cfg.still_radius:
                break
            start = ts
        return start

    def _advance(self, hist: _History, t: float, speed: float | None) -> None:
        cfg = self.cfg
        if speed is None:
            return
        if speed < cfg.stopped_below:
            if hist.still_since is None:
                hist.still_since = hist.samples[0][0] if hist.stationary_from_start else self._rest_start(hist)
            if not hist.stopped and t - hist.still_since >= cfg.stopped_after_s:
                hist.stopped = True
                _, x, y, h = hist.samples[-1]
                hist.stop_anchor = (x, y, h)
        elif speed > cfg.moving_above:
            hist.seen_moving = True
            hist.stationary_from_start = False
            hist.stopped = False
            hist.still_since = None
            # stop_anchor is kept: moved_from_stop keeps growing as the car drives off (departure detection);
            # it is replaced when the car comes to rest somewhere else.
        elif not hist.stopped:
            hist.still_since = None
            hist.stationary_from_start = False
