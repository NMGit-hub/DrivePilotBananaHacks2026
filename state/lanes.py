"""Our lane, the lanes beside it, and the drift warning, from YOLOPv2's masks (V2-M5).

Geometry. A pinhole camera at height h above a flat road, horizon at image row y_h, principal point at the image
center: a road point seen at pixel (x, y) below the horizon lies Z = f·h / (y − y_h) ahead and X = (x − cx)·h /
(y − y_h) to the side. The lateral position needs only h and the horizon (not the focal length), and h is only a
prior: our own lane's measured width sets the metric scale (lanes are `lane_width_m` wide on average), so a wrong
camera height or a different car doesn't matter. The horizon is measured: the lane lines' near parts are fitted
in the image and their common vanishing point is tracked (slowly, starting from `horizon_prior`).

Lines. Every row of the lane-line mask gives the centers of its runs; in the top-down view all lane lines of a
road are parallel curves X = X0 + a·Z + b·Z² (shared heading a and curvature b, their own offset X0). The (a, b)
that make the offsets X − a·Z − b·Z² of all line pixels pile up into the sharpest histogram is searched (coarse to
fine); the histogram's peaks are the lines, X0 their positions beside the car. This copes with dashed lines,
lines hidden by cars over part of their length, and curves.

Lanes. Our lane is between the nearest line on each side (its width checked). Lanes to the left are counted line
by line while each gap is one lane wide (`gap_min`..`gap_max` of our lane), up to the drivable area's left edge
(YOLOPv2's drivable area covers the lanes and shoulders of our direction); likewise to the right. A side whose
edge isn't seen, or whose gaps aren't lane-wide, gives None. The count and our index (from the left) are the
majority over the last `vote_window_s`; when it isn't clear, they are None, so callers fall back to the route data
or the vision model.

Drift. Our offset from the lane center is unwrapped across lane changes (a continuous lateral position), smoothed,
and its speed fitted over `velocity_window_s`. The offset is measured from the camera, and neither where the camera
sits across the windshield nor where this driver keeps the car in the lane is known (on the BDD100K highway clips
the median offset is −0.5..+0.2 m), so drift is measured from our usual place: the median offset over the last
`habit_window_s` of driving at speed (none before `habit_min_s` of it: no alerts until then). A drift is closing in
on a line slowly and steadily (at least `drift_min_mps`, slower than `lane_change_mps`, `drift_min_shift_m` over
`drift_window_s`, moving toward it in `drift_steady_share` of that window) until the car's side, counted from our
usual place, is `drift_gap_m` from it, at `min_speed_mps` (25 mph) or more by GPS. A fast sideways move is a lane
change (we can't see the turn signal): no alert, and none for `lane_change_quiet_s` after crossing a line. The
steadiness test matters at a lane change's start: the fitted speed lags the real one by most of
`velocity_window_s`, so the first second of a fast move looks slow; the speed over the last `quick_window_s` must
not be a lane change's either. One alert per drift: re-armed when we're back
near our usual place or change lanes, with a cooldown.

Called from the lanes thread (`update`, which measures outside the lock); `snapshot` and `reset` are safe from any
thread (a reset is applied by the next `update`).
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from core.config import LanesConfig

Side = Literal["left", "right"]
Point = tuple[float, float]

# Search grids for the shared heading a and curvature b (coarse, then fine around the best coarse cell).
_COARSE_A_STEP = 0.02
_COARSE_B_STEP = 0.001
_FINE_STEPS = 4  # fine cells per coarse cell...
_FINE_SPAN = 2  # ...searched this many coarse cells around the coarse best (heading and curvature trade off)
_SEARCH_BIN_M = 0.2  # coarser bins for the coarse search (robust); peaks use cfg.bin_m
_FINE_BIN_M = 0.05  # bins for the fine search
_X_RANGE_M = 16.0  # line offsets considered: within this of the car (before scaling)
_MEMBER_M = 0.3  # a pixel within this of a line's offset belongs to it
_HORIZON_FAST_SAMPLES = 10  # until this many vanishing rows are measured, the estimate is their median
_HORIZON_WINDOW = 90  # the estimate follows the median of the last this many (~7 s at 12 fps)
_VP_MAX_RUN = 40  # mask px: wider runs (stop lines, blobs) don't give a direction
_VP_LINK_ROWS = 3  # segments link run centers this many mask rows apart...
_VP_LINK_MAX = 30  # ...at most this many mask px apart sideways
_VP_MIN_LEAN = 0.3  # segments leaning at least this much (dx/dy) on both sides constrain the row
_VP_ROW_STEP = 4.0  # frame px between candidate rows
_VP_BIN_PX = 10.0  # frame px: crossing points within ~3 bins agree
_VP_MIN_SCORE = 0.2  # share of each side's segments that must cross near one point
_DRIVABLE_FAR_M = 20.0  # the drivable area's edges are read this far ahead
_DRIVABLE_MERGE_M = 2.0  # drivable runs separated by less (a car beside us) are one road
_EDGE_PERCENTILE = 80.0  # the drivable area's reach: this percentile over rows (cars hide some rows)
_FAN_MEMBER_M = 0.4  # line pixels this near a line's offset are fitted for its fan-out slope...
_FAN_MIN_RANGE_M = 6.0  # ...if they span this much distance
_FAN_MAX_STEP = 0.03  # one frame's fan-out measurement moves the horizon at most this (height share)
_POLY_STEP_M = 2.0
_HABIT_MAX_GAP_S = 0.5  # a habit sample stands for at most this long (frames missing in between don't count)
_LANE_CHANGE_CONFIRM_S = 1.0  # a lane change is logged once we've stayed in the new lane this long


class MasksLike(Protocol):
    """What the tracker reads from the lane model (perception.lanes.LaneMasks has these)."""

    @property
    def lines(self) -> np.ndarray: ...
    @property
    def drivable(self) -> np.ndarray: ...
    @property
    def frame_size(self) -> tuple[int, int]: ...


@dataclass(frozen=True, slots=True)
class LaneLine:
    """One lane line in one frame (offsets in metres, + = right of the car's center line)."""

    offset_m: float  # beside the car (Z = 0)
    support_m: float  # how much of its length was detected
    near_m: float  # detected from this far ahead...
    far_m: float  # ...to this far (unscaled)
    points: tuple[Point, ...]  # polyline in normalized frame coordinates (0..1), near to far


@dataclass(frozen=True, slots=True)
class LaneMeasurement:
    """What one frame shows (before smoothing over time)."""

    horizon: float  # frame-height share used for this frame
    heading: float  # dX/dZ shared by the lines
    curvature: float  # dX/dZ² shared by the lines
    lines: tuple[LaneLine, ...]  # left to right
    left: int | None  # index into `lines` of our lane's left line
    right: int | None  # ...and right line
    width_m: float | None  # our lane's width (scaled), if both lines are seen and it is plausible
    offset_m: float | None  # car center from our lane's center (scaled, + = right)
    lanes_left: int | None  # lanes to our left (None: not clear)
    lanes_right: int | None
    drivable_left_m: float | None  # the drivable area's edges (scaled offsets), None if not seen
    drivable_right_m: float | None


@dataclass(frozen=True, slots=True)
class DriftEvent:
    """Something to say: we're drifting toward a line."""

    t: float
    side: Side
    text: str
    gap_m: float  # car side to the line (counted from our usual place in the lane)
    lateral_speed_mps: float  # toward the line
    offset_m: float  # car center from the lane center (+ = right)
    habit_m: float  # our usual offset from the lane center
    speed_mps: float  # GPS speed


def drift_text(side: Side) -> str:
    """The spoken alert (PLAN.md feature 2)."""
    return f"Careful, you're drifting {side}."


DRIFT_TEXTS = (drift_text("left"), drift_text("right"))


@dataclass(frozen=True, slots=True)
class LineView:
    """A lane line for the page."""

    points: tuple[Point, ...]  # normalized frame coordinates, near to far
    offset_m: float
    role: Literal["left", "right", "other"]  # our lane's left/right line, or another lane's


@dataclass(frozen=True, slots=True)
class LanesSnapshot:
    """What the page and the answers need: our lane, the lines, lane count and index, drift."""

    t: float
    valid: bool  # our lane is seen in the latest frame
    lane_polygon: tuple[Point, ...]  # our lane (normalized frame coordinates): left line near→far, right far→near
    lines: tuple[LineView, ...]  # left to right
    drift_side: Side | None  # the line we're drifting toward (from the alert until re-armed)
    drift_line: tuple[Point, ...] | None  # ...its polyline
    lane_count: int | None  # lanes in our direction (None: not clear)
    lane_index: int | None  # ours, 1 = leftmost (None: not clear)
    lanes_left: int | None  # lanes to our left / right (each may be known without the other)
    lanes_right: int | None
    confidence: float  # 0..1: agreement of the recent frames on lanes_left/lanes_right
    offset_m: float | None  # car center from the lane center (+ = right)
    lane_width_m: float | None
    lateral_speed_mps: float | None  # + = moving right
    horizon: float
    habit_m: float | None = None  # our usual offset from the lane center (None: not learned yet)


EMPTY = LanesSnapshot(0.0, False, (), (), None, None, None, None, None, None, 0.0, None, None, None, 0.5)


@dataclass(frozen=True, slots=True)
class _SideView:
    """One side of our lane, row by row (unscaled offsets at Z = 0, + = outward on this side).

    `reach`: how far the drivable area around our lane goes in each row (a row cut off by the image border
    reaches at least the border); `view`: how far the camera sees in that row (the image border).
    """

    reach: np.ndarray
    view: np.ndarray

    def beyond(self, offset: float, slack: float) -> float | None:
        """How far past an (outward) offset the drivable area goes, from the rows that see past it by `slack`.

        A high percentile over those rows (a car beside us hides the road behind it in some rows); None if fewer
        than 3 rows see that far.
        """
        seeing = self.view >= offset + slack
        if int(seeing.sum()) < 3:
            return None
        return float(np.percentile(self.reach[seeing], _EDGE_PERCENTILE)) - offset

    @property
    def edge(self) -> float:
        """The drivable area's typical reach (for reporting)."""
        return float(np.percentile(self.reach, _EDGE_PERCENTILE))


@dataclass(slots=True)
class _Geometry:
    """Pinhole projection for one frame size and horizon (unscaled metres: camera height = the prior)."""

    width: int
    height: int
    f: float  # focal length in frame pixels
    horizon_px: float
    camera_height: float

    @property
    def cx(self) -> float:
        return self.width / 2.0

    def to_ground(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(X, Z) of image points below the horizon."""
        dy = y - self.horizon_px
        return (x - self.cx) * self.camera_height / dy, self.f * self.camera_height / dy

    def to_image(self, x_m: np.ndarray, z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Image (x, y) of ground points."""
        dy = self.f * self.camera_height / z_m
        return self.cx + x_m * dy / self.camera_height, self.horizon_px + dy

    def row_for_z(self, z_m: float) -> float:
        """Image row of a ground distance."""
        return self.horizon_px + self.f * self.camera_height / z_m


def _runs(mask: np.ndarray, r0: int, r1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(rows, start cols, end cols (exclusive)) of the horizontal runs of 1s in mask rows r0..r1, plus run widths."""
    band = mask[r0:r1].astype(np.int8)
    if band.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    padded = np.pad(band, ((0, 0), (1, 1)))
    d = np.diff(padded, axis=1)
    sr, sc = np.nonzero(d == 1)
    _, ec = np.nonzero(d == -1)  # same order as the starts (row-major, alternating)
    return sr + r0, sc, ec, ec - sc


def _histogram(off: np.ndarray, w: np.ndarray, lo: float, bin_m: float, nbins: int) -> np.ndarray:
    """Weighted histogram with linear interpolation between bins (no aliasing at bin edges)."""
    idx = (off - lo) / bin_m
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    ok = (i0 >= 0) & (i0 < nbins - 1)
    i0, frac, w = i0[ok], frac[ok], w[ok]
    hist = np.bincount(i0, weights=w * (1.0 - frac), minlength=nbins)
    hist += np.bincount(i0 + 1, weights=w * frac, minlength=nbins)
    return hist[:nbins]


def _search(
    x: np.ndarray, z: np.ndarray, w: np.ndarray, a_grid: np.ndarray, b_grid: np.ndarray, bin_m: float
) -> tuple[float, float, float]:
    """The (heading, curvature) whose offsets are most concentrated, and that concentration.

    The histograms are smoothed with a binomial kernel (σ = 1 bin) before scoring: on raw (linearly interpolated)
    bins a line at a bin center scores twice as sharp as one on a bin edge, so the search would prefer a slightly
    wrong heading and curvature that move the lines onto bin centers (measured: lines 0.15 m off).
    """
    nbins = int(2 * _X_RANGE_M / bin_m) + 2
    na, nb = len(a_grid), len(b_grid)
    off = x[None, None, :] - a_grid[:, None, None] * z[None, None, :] - b_grid[None, :, None] * (z * z)[None, None, :]
    idx = (off + _X_RANGE_M) / bin_m
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    ok = (i0 >= 0) & (i0 < nbins - 1)
    combo = np.arange(na * nb, dtype=np.int64).reshape(na, nb, 1) * nbins
    flat = np.broadcast_to(combo, off.shape)[ok] + i0[ok]
    ww = np.broadcast_to(w[None, None, :], off.shape)[ok]
    fr = frac[ok]
    size = na * nb * nbins
    hist = np.bincount(flat, weights=ww * (1.0 - fr), minlength=size)[:size]
    hist += np.bincount(flat + 1, weights=ww * fr, minlength=size + 1)[:size]
    hist = hist.reshape(na * nb, nbins)
    padded = np.pad(hist, ((0, 0), (2, 2)))
    smooth = (padded[:, :-4] + 4 * padded[:, 1:-3] + 6 * padded[:, 2:-2] + 4 * padded[:, 3:-1] + padded[:, 4:]) / 16.0
    total = smooth.sum(axis=1)
    score = (smooth * smooth).sum(axis=1) / np.maximum(total * total, 1e-12)
    best = int(np.argmax(score))
    return float(a_grid[best // nb]), float(b_grid[best % nb]), float(score[best])


def vanishing_row(lines: np.ndarray, size: tuple[int, int], lo: float, hi: float, hood: float) -> float | None:
    """The row (frame-height share) where this frame's lane lines meet, or None if they don't show one.

    Short segments of the lines (run centers linked over a few rows) are extended upward; at the true horizon
    row the segments of every line meet (one point on a straight road; nearby points on a curve), at a wrong
    row they spread out. The row whose crossing points pile up the most wins (a Hough vote over rows).
    Needs lines leaning both ways (left and right of the vanishing point), else the row is not constrained.
    """
    mh, mw = lines.shape
    w, h = size
    sx, sy = w / mw, h / mh
    r0, r1 = max(0, int(lo * mh)), min(mh, int(hood * mh))
    rows, c0, c1, widths = _runs(lines, r0, r1)
    keep = widths <= _VP_MAX_RUN
    rows, centers = rows[keep], (c0[keep] + c1[keep] - 1) / 2.0
    if len(rows) < 20:
        return None
    by_row: dict[int, np.ndarray] = {}
    for r in np.unique(rows):
        by_row[int(r)] = centers[rows == r]
    xs, ys, betas = [], [], []
    for r, cs in by_row.items():
        below = by_row.get(r + _VP_LINK_ROWS)
        if below is None:
            continue
        nearest = np.abs(cs[:, None] - below[None, :])
        j = nearest.argmin(axis=1)
        ok = nearest[np.arange(len(cs)), j] <= _VP_LINK_MAX
        for c, cb in zip(cs[ok], below[j[ok]], strict=True):
            xs.append((c + 0.5) * sx)
            ys.append((r + 0.5) * sy)
            betas.append((cb - c) * sx / (_VP_LINK_ROWS * sy))
    if len(xs) < 20:
        return None
    x = np.array(xs)
    y = np.array(ys)
    beta = np.array(betas)
    if (beta < -_VP_MIN_LEAN).sum() < 5 or (beta > _VP_MIN_LEAN).sum() < 5:
        return None
    candidates = np.arange(lo * h, hi * h, _VP_ROW_STEP)
    nb = int(3 * w / _VP_BIN_PX) + 1
    # Every segment's line crosses every candidate row (extended up or down): the same segments vote everywhere.
    xv = x[None, :] + beta[None, :] * (candidates[:, None] - y[None, :])
    idx = np.floor((xv + w) / _VP_BIN_PX).astype(np.int64)
    ok = (idx >= 0) & (idx < nb)
    cell = np.arange(len(candidates))[:, None] * nb + idx
    shares = []
    # Lines leaning left and lines leaning right vote separately, and a point scores the smaller of the two shares
    # passing through it: segments of one curved line also cross each other (near the line itself), which must
    # not outvote the point where the lines on both sides meet.
    for family in (beta < 0.0, beta > 0.0):
        sel = ok & family[None, :]
        hist = np.bincount(cell[sel], minlength=len(candidates) * nb).reshape(len(candidates), nb).astype(float)
        hist[:, 1:-1] += hist[:, :-2] + hist[:, 2:]  # a 3-bin window (the right-hand side is evaluated first)
        shares.append(hist / max(1, int(family.sum())))
    score = np.minimum(shares[0], shares[1]).max(axis=1)
    best = int(np.argmax(score))
    if score[best] < _VP_MIN_SCORE:
        return None
    return float(candidates[best]) / h


def _grid(center: float, half: float, step: float) -> np.ndarray:
    n = max(1, round(half / step))
    return center + step * np.arange(-n, n + 1)


class LaneTracker:
    """Lane masks in; our lane, lane count/index and drift events out."""

    def __init__(self, cfg: LanesConfig, hfov_deg: float) -> None:
        self.cfg = cfg
        self.hfov_deg = hfov_deg
        self._lock = threading.Lock()
        # The camera's horizon and the metric scale belong to the mount, not the moment: reset() keeps them.
        self._horizon = cfg.horizon_prior
        self._horizon_n = 0
        self._vanishing: deque[float] = deque(maxlen=_HORIZON_WINDOW)  # recent measured vanishing rows
        self._widths: deque[float] = deque(maxlen=cfg.scale_window)
        # So does our usual place in the lane (where the camera sits, how this driver keeps the lane).
        self._habit: deque[tuple[float, float, float]] = deque()  # (t, offset, seconds it stands for)
        self._habit_s = 0.0  # seconds of driving the samples stand for
        self._habit_value: float | None = None
        self.lane_changes: deque[tuple[float, int]] = deque(maxlen=64)  # (t, +1 right / -1 left), for logs
        self._reset_pending = False
        self._snapshot = EMPTY
        self.last: LaneMeasurement | None = None
        # Tracking state (set by _clear / _forget_track).
        self._last_t: float | None = None
        self._last_valid_t: float | None = None
        self._k_since: float | None = None  # when _k last changed
        self._p_prev: float | None = None
        self._p_smooth: float | None = None
        self._positions: deque[tuple[float, float]] = deque()  # (t, unwrapped raw position)
        self._speeds: deque[tuple[float, float]] = deque()  # (t, fitted lateral speed)
        self._votes: deque[tuple[float, int | None, int | None]] = deque()
        self._drift_side: Side | None = None
        self._clear()

    def reset(self) -> None:
        """Forget the tracked position, votes and drift state (a seek, a new segment; any thread). Applied by the
        next `update`; the snapshot is empty until then."""
        with self._lock:
            self._reset_pending = True
            self._snapshot = EMPTY

    def _clear(self) -> None:
        """The reset itself (lanes thread)."""
        self._last_t = None
        self._last_lane_change_t = -math.inf
        self._last_alert_t = -math.inf
        self.last = None
        self._forget_track()

    # ------------------------------------------------------------------------------------------ geometry

    @property
    def horizon(self) -> float:
        """The current horizon estimate (frame-height share)."""
        return self._horizon

    @property
    def scale(self) -> float:
        """Metric scale: nominal lane width / the median measured (unscaled) width of our lane, clamped."""
        cfg = self.cfg
        if not self._widths:
            return 1.0
        return float(np.clip(cfg.lane_width_m / float(np.median(self._widths)), cfg.scale_min, cfg.scale_max))

    def _geometry(self, size: tuple[int, int], horizon: float) -> _Geometry:
        w, h = size
        f = (w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        return _Geometry(w, h, f, horizon * h, self.cfg.camera_height_m)

    # ------------------------------------------------------------------------------------------ measure

    def measure(self, masks: MasksLike) -> LaneMeasurement:
        """One frame's lines, lanes and position (uses and updates the horizon and scale estimates)."""
        cfg = self.cfg
        self._update_horizon(masks.lines, masks.frame_size)
        geo = self._geometry(masks.frame_size, self._horizon)
        mh, mw = masks.lines.shape
        sx, sy = geo.width / mw, geo.height / mh
        empty = LaneMeasurement(self._horizon, 0.0, 0.0, (), None, None, None, None, None, None, None, None)

        # Run centers of the lane-line mask between far_m and the hood, in the top-down view.
        r0 = max(0, math.ceil(geo.row_for_z(cfg.far_m) / sy - 0.5))
        r1 = min(mh, math.floor(min(geo.row_for_z(cfg.near_m), cfg.hood_cutoff * geo.height) / sy))
        if r1 - r0 < 4:
            return empty
        rows, c0, c1, widths = _runs(masks.lines, r0, r1)
        if len(rows) < 8:
            return empty
        xi = (c0 + c1) / 2.0 * sx
        yi = (rows + 0.5) * sy
        x_m, z_m = geo.to_ground(xi, yi)
        width_m = widths * sx * cfg.camera_height_m / (yi - geo.horizon_px)
        keep = (width_m <= cfg.max_line_width_m * (1.0 + np.abs(x_m) / cfg.camera_height_m)) & (
            np.abs(x_m) < _X_RANGE_M
        )
        xi, yi, x_m, z_m = xi[keep], yi[keep], x_m[keep], z_m[keep]
        if len(x_m) < 8:
            return empty
        weight = z_m / cfg.near_m  # rows thin out with distance: far pixels count more

        # Shared heading and curvature: coarse grid, then fine around the best cell.
        a_c = _grid(0.0, cfg.max_heading, _COARSE_A_STEP)
        b_c = _grid(0.0, cfg.max_curvature, _COARSE_B_STEP) if cfg.max_curvature > 0 else np.zeros(1)
        a0, b0, _ = _search(x_m, z_m, weight, a_c, b_c, _SEARCH_BIN_M)
        a_f = _grid(a0, _FINE_SPAN * _COARSE_A_STEP, _COARSE_A_STEP / _FINE_STEPS)
        b_f = np.zeros(1)
        if cfg.max_curvature > 0:
            b_f = _grid(b0, _FINE_SPAN * _COARSE_B_STEP, _COARSE_B_STEP / _FINE_STEPS)
        heading, curvature, _ = _search(x_m, z_m, weight, a_f, b_f, _FINE_BIN_M)

        lines = self._find_lines(geo, x_m, z_m, weight, heading, curvature)
        self._fan_out(geo, lines, x_m, z_m, heading, curvature)
        return self._lanes(geo, masks, lines, heading, curvature)

    def _fan_out(
        self, geo: _Geometry, lines: list[LaneLine], x_m: np.ndarray, z_m: np.ndarray, heading: float, curvature: float
    ) -> None:
        """Refine the horizon from how the lines fan out with distance.

        With the horizon δ pixels too low, every offset reads X·(1 + δ·Z/(f·h)): a line's residual offset grows
        with distance in proportion to its offset. Regressing the lines' residual slopes on their offsets (with
        an intercept, which the shared heading absorbs) gives δ. Outer lanes are only seen far away, so their
        offsets depend on this most. Needs lines spread across both sides of the car.
        """
        cfg = self.cfg
        if self._horizon_n < _HORIZON_FAST_SAMPLES:
            return  # the vanishing row first brings the estimate near
        off = x_m - heading * z_m - curvature * z_m * z_m
        x0s, slopes, ws = [], [], []
        for line in lines:
            sel = np.abs(off - line.offset_m) < _FAN_MEMBER_M
            if int(sel.sum()) < 10 or float(np.ptp(z_m[sel])) < _FAN_MIN_RANGE_M:
                continue
            slope = float(np.polyfit(z_m[sel], off[sel], 1)[0])
            x0s.append(line.offset_m)
            slopes.append(slope)
            ws.append(float(sel.sum()))
        if len(x0s) < 2 or min(x0s) > -1.0 or max(x0s) < 1.0:
            return
        x = np.array(x0s)
        s = np.array(slopes)
        w = np.array(ws)
        xm = np.average(x, weights=w)
        var = float(np.average((x - xm) ** 2, weights=w))
        if var < 1.0:
            return
        k = float(np.average((x - xm) * (s - np.average(s, weights=w)), weights=w)) / var
        delta_px = float(
            np.clip(k * geo.f * geo.camera_height, -_FAN_MAX_STEP * geo.height, _FAN_MAX_STEP * geo.height)
        )
        measured = (geo.horizon_px - delta_px) / geo.height
        if cfg.horizon_min <= measured <= cfg.horizon_max:
            self._horizon += cfg.horizon_rate * (measured - self._horizon)

    def _find_lines(
        self,
        geo: _Geometry,
        x_m: np.ndarray,
        z_m: np.ndarray,
        weight: np.ndarray,
        heading: float,
        curvature: float,
    ) -> list[LaneLine]:
        """Peaks of the offset histogram at the best heading/curvature, as lines (left to right)."""
        cfg = self.cfg
        off = x_m - heading * z_m - curvature * z_m * z_m
        nbins = int(2 * _X_RANGE_M / cfg.bin_m) + 2
        hist = _histogram(off, weight, -_X_RANGE_M, cfg.bin_m, nbins)
        sigma = cfg.line_sigma_m / cfg.bin_m
        radius = max(1, math.ceil(3 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        smooth = np.convolve(hist, kernel / kernel.sum(), mode="same")
        order = np.argsort(-smooth)
        taken: list[float] = []
        found: list[LaneLine] = []
        sep = cfg.min_line_sep_m
        for i in order:
            if smooth[i] <= 0:
                break
            center = -_X_RANGE_M + i * cfg.bin_m
            if any(abs(center - c) < sep for c in taken):
                continue
            if i == 0 or i == nbins - 1 or smooth[i] < smooth[i - 1] or smooth[i] < smooth[i + 1]:
                continue  # not a local maximum
            member = np.abs(off - center) < _MEMBER_M
            if member.sum() < 4:
                continue
            zs = z_m[member]
            support = float(len(np.unique(np.floor(zs))))  # metres of Z (1 m bins) with line pixels
            if support < cfg.min_line_support_m:
                continue
            taken.append(center)
            x0 = float(np.average(off[member], weights=weight[member]))
            near, far = float(zs.min()), float(zs.max())
            found.append(LaneLine(x0, support, near, far, self._polyline(geo, x0, heading, curvature, near, far)))
            if len(found) >= 12:
                break
        found.sort(key=lambda line: line.offset_m)
        return found

    def _polyline(
        self, geo: _Geometry, x0: float, heading: float, curvature: float, near: float, far: float
    ) -> tuple[Point, ...]:
        """The line X0 + aZ + bZ² from `near` to `far`, in normalized frame coordinates (unscaled X0)."""
        n = max(2, math.ceil((far - near) / _POLY_STEP_M) + 1)
        zs = np.linspace(near, far, n)
        xs, ys = geo.to_image(x0 + heading * zs + curvature * zs * zs, zs)
        return tuple(
            (round(float(x) / geo.width, 4), round(float(y) / geo.height, 4)) for x, y in zip(xs, ys, strict=True)
        )

    def _update_horizon(self, lines_mask: np.ndarray, size: tuple[int, int]) -> None:
        """Move the horizon estimate toward this frame's vanishing row (if the frame shows one)."""
        cfg = self.cfg
        measured = vanishing_row(lines_mask, size, cfg.horizon_min, cfg.horizon_max, cfg.hood_cutoff)
        if measured is None:
            return
        # The median of the recent rows, not single ones: on a curve, at a fork or behind a truck a frame's
        # crossing points can pile up at a wrong row. No outlier gate against the estimate: if the drive started on
        # a long curve, the estimate is off, and a gate would then keep out the straight road's right rows for good.
        self._vanishing.append(measured)
        self._horizon_n += 1
        target = float(np.median(self._vanishing))
        if self._horizon_n <= _HORIZON_FAST_SAMPLES:
            self._horizon = target
        else:
            self._horizon += cfg.horizon_rate * (target - self._horizon)

    def _drivable_sides(
        self, geo: _Geometry, drivable: np.ndarray, center_m: float, heading: float, curvature: float
    ) -> tuple[_SideView | None, _SideView | None]:
        """How far the drivable area reaches on each side of our lane, and how far the camera sees there.

        Rows from near_m to _DRIVABLE_FAR_M are read in the top-down view (offsets with the lines' shared curve
        taken out). Per row, the drivable run around our lane (gaps narrower than _DRIVABLE_MERGE_M merged) gives
        its reach; a run cut off by the image border reaches at least the border. Cars beside us hide the road
        behind them in some rows, so the reach is a high percentile over the rows, not the median.
        """
        cfg = self.cfg
        mh, mw = drivable.shape
        sx, sy = geo.width / mw, geo.height / mh
        r0 = max(0, math.ceil(geo.row_for_z(cfg.far_m) / sy))
        r1 = min(mh, math.floor(min(geo.row_for_z(cfg.near_m), cfg.hood_cutoff * geo.height) / sy))
        reach: dict[int, list[float]] = {-1: [], 1: []}
        border: dict[int, list[float]] = {-1: [], 1: []}
        for r in range(r0, r1, 2):
            dy = (r + 0.5) * sy - geo.horizon_px
            cols = np.flatnonzero(drivable[r])
            if dy <= 0 or len(cols) == 0:
                continue
            z = geo.f * geo.camera_height / dy
            shift = heading * z + curvature * z * z
            to_x = geo.camera_height / dy
            xs = ((cols + 0.5) * sx - geo.cx) * to_x - shift  # offsets of the drivable pixels
            gaps = np.flatnonzero(np.diff(xs) > _DRIVABLE_MERGE_M)
            starts = np.concatenate(([0], gaps + 1))
            ends = np.concatenate((gaps, [len(xs) - 1]))
            dist = np.where(
                (xs[starts] <= center_m) & (center_m <= xs[ends]),
                0.0,
                np.minimum(np.abs(xs[starts] - center_m), np.abs(xs[ends] - center_m)),
            )
            k = int(np.argmin(dist))
            left_border = (0.5 * sx - geo.cx) * to_x - shift
            right_border = ((mw - 0.5) * sx - geo.cx) * to_x - shift
            reach[-1].append(left_border if cols[starts[k]] <= 1 else float(xs[starts[k]]))
            reach[1].append(right_border if cols[ends[k]] >= mw - 2 else float(xs[ends[k]]))
            border[-1].append(left_border)
            border[1].append(right_border)
        out: list[_SideView | None] = []
        for side in (-1, 1):  # stored outward-positive: the left side's offsets are negated
            if len(reach[side]) < 3:
                out.append(None)
                continue
            out.append(_SideView(side * np.array(reach[side]), side * np.array(border[side])))
        return out[0], out[1]

    def _lanes(
        self, geo: _Geometry, masks: MasksLike, lines: list[LaneLine], heading: float, curvature: float
    ) -> LaneMeasurement:
        """Our lane, its width and our offset, and the lanes on each side."""
        cfg = self.cfg
        car = -cfg.camera_offset_m / self.scale  # the car's center line (unscaled, camera coordinates)
        left_side = [i for i, ln in enumerate(lines) if ln.offset_m < car]  # lines are sorted left to right
        right_side = [i for i, ln in enumerate(lines) if ln.offset_m >= car]
        left = left_side[-1] if left_side else None
        right = right_side[0] if right_side else None
        base = LaneMeasurement(
            self._horizon, heading, curvature, tuple(lines), left, right, None, None, None, None, None, None
        )
        if left is None or right is None:
            return base
        width_u = lines[right].offset_m - lines[left].offset_m
        lo, hi = cfg.lane_width_m / cfg.scale_max, cfg.lane_width_m / cfg.scale_min
        if not lo <= width_u <= hi:
            return base
        if self._widths:
            ratio = width_u / float(np.median(self._widths))
            if not cfg.lane_width_min_m / cfg.lane_width_m <= ratio <= cfg.lane_width_max_m / cfg.lane_width_m:
                return base
        self._widths.append(width_u)
        scale = self.scale
        center_u = (lines[left].offset_m + lines[right].offset_m) / 2.0
        v_left, v_right = self._drivable_sides(geo, masks.drivable, center_u, heading, curvature)
        lanes_left = self._count_side(lines, left, -1, width_u, v_left)
        lanes_right = self._count_side(lines, right, +1, width_u, v_right)
        d_left = None if v_left is None else -v_left.edge
        d_right = None if v_right is None else v_right.edge
        return LaneMeasurement(
            horizon=self._horizon,
            heading=heading,
            curvature=curvature,
            lines=tuple(lines),
            left=left,
            right=right,
            width_m=width_u * scale,
            offset_m=(car - center_u) * scale,
            lanes_left=lanes_left,
            lanes_right=lanes_right,
            drivable_left_m=None if d_left is None else d_left * scale,
            drivable_right_m=None if d_right is None else d_right * scale,
        )

    def _count_side(
        self, lines: list[LaneLine], start: int, step: int, width_u: float, view: _SideView | None
    ) -> int | None:
        """Lanes beyond our lane's line `start` on one side (step -1: left, +1: right), or None if not clear.

        Walks outward line by line while each gap is one lane wide and inside the drivable area. It stops at a
        line the drivable area ends at (or at a shoulder narrower than a lane), and only if the camera sees past
        that edge: otherwise the road may go on out of view, and the answer is None.
        """
        cfg = self.cfg
        if view is None:
            return None
        slack = cfg.edge_drivable_max * width_u
        count = 0
        i = start
        while True:
            here = step * lines[i].offset_m  # outward-positive, like the view
            j = i + step
            gap = abs(lines[j].offset_m - lines[i].offset_m) if 0 <= j < len(lines) else None
            if gap is not None and gap <= cfg.gap_max * width_u:
                reach = view.beyond(here, min(gap, slack))  # does the road reach the next line?
                if reach is not None and reach >= gap - slack:
                    if gap >= cfg.gap_min * width_u:
                        count += 1  # one more lane
                    i = j  # (a line nearer than a lane inside the road: a double line, a shoulder's edge)
                    continue
            beyond = view.beyond(here, slack)
            if beyond is None:
                return None  # we can't see past this line: the road may go on
            if beyond <= slack or (gap is None and beyond < cfg.gap_min * width_u):
                return count  # the road's edge (with at most a shoulder narrower than a lane past it)
            return None

    # -------------------------------------------------------------------------------------------- track

    def update(self, t: float, masks: MasksLike, gps_speed_mps: float | None) -> list[DriftEvent]:
        """One frame's masks (lanes thread only); returns drift alerts to speak. Measures outside the lock, so
        `snapshot()` never waits for a measurement."""
        cfg = self.cfg
        with self._lock:
            pending, self._reset_pending = self._reset_pending, False
        if pending:
            self._clear()
        if self._last_t is not None and t < self._last_t:
            self._forget_track()
        self._last_t = t
        m = self.measure(masks)
        self.last = m
        events: list[DriftEvent] = []
        if m.offset_m is not None and m.width_m is not None:
            if self._last_valid_t is not None and t - self._last_valid_t > cfg.lost_s:
                self._forget_track()
            self._last_valid_t = t
            events = self._track(t, m, gps_speed_mps)
        elif self._last_valid_t is not None and t - self._last_valid_t > cfg.lost_s:
            self._forget_track()
        self._votes.append((t, m.lanes_left, m.lanes_right))
        while self._votes and t - self._votes[0][0] > cfg.vote_window_s:
            self._votes.popleft()
        snapshot = self._make_snapshot(t, m)
        with self._lock:
            if self._reset_pending:
                return []  # a reset came in meanwhile: this frame belongs to before it
            self._snapshot = snapshot
        return events

    def _forget_track(self) -> None:
        """Forget the tracked lateral position and votes (lost lanes, time going backwards)."""
        self._k = 0
        self._k_logged = 0  # the lane we were last confirmed in (lane changes in the log)
        self._k_since = None
        self._p_prev = None
        self._p_smooth = None
        self._positions = deque()
        self._speeds = deque()
        self._votes = deque()
        self._last_valid_t = None
        self._drift_side = None
        self._armed = {"left": True, "right": True}

    def _learn_habit(self, t: float, offset: float) -> None:
        """One sample of where in the lane we drive (at speed, not around a lane change)."""
        cfg = self.cfg
        credit = min(t - self._habit[-1][0], _HABIT_MAX_GAP_S) if self._habit else 0.0
        self._habit.append((t, offset, max(0.0, credit)))
        self._habit_s += max(0.0, credit)
        while self._habit and t - self._habit[0][0] > cfg.habit_window_s:
            self._habit_s -= self._habit.popleft()[2]
        if self._habit_s >= cfg.habit_min_s:
            self._habit_value = float(np.median(np.fromiter((s[1] for s in self._habit), float, len(self._habit))))
        else:
            self._habit_value = None

    def _track(self, t: float, m: LaneMeasurement, speed: float | None) -> list[DriftEvent]:
        """Unwrap, smooth and differentiate our lateral position; decide on a drift alert."""
        cfg = self.cfg
        assert m.offset_m is not None and m.width_m is not None
        width = m.width_m
        d = m.offset_m
        # Unwrap across lane changes: the lane-relative offset jumps by a lane width when we cross a line.
        if self._p_prev is not None:
            best_k = min((self._k - 1, self._k, self._k + 1), key=lambda k: abs(d + k * width - self._p_prev))  # type: ignore[operator]
            if best_k != self._k:
                self._last_lane_change_t = t
                self._k = best_k
                self._k_since = t
                self._armed = {"left": True, "right": True}
                self._drift_side = None
        if self._k != self._k_logged and self._k_since is not None and t - self._k_since >= _LANE_CHANGE_CONFIRM_S:
            self.lane_changes.append((self._k_since, self._k - self._k_logged))  # (flip-flops are not logged)
            self._k_logged = self._k
        p = d + self._k * width
        dt = t - self._positions[-1][0] if self._positions else 0.0
        self._p_prev = p
        alpha = 1.0 if self._p_smooth is None else 1.0 - math.exp(-max(dt, 0.0) / cfg.smooth_s)
        self._p_smooth = p if self._p_smooth is None else self._p_smooth + alpha * (p - self._p_smooth)
        self._positions.append((t, p))
        horizon_s = max(cfg.velocity_window_s, cfg.drift_window_s) + 1.0
        while self._positions and t - self._positions[0][0] > horizon_s:
            self._positions.popleft()
        v = self._velocity(t)
        if v is not None:
            self._speeds.append((t, v))
        while self._speeds and t - self._speeds[0][0] > horizon_s:
            self._speeds.popleft()

        offset = self._p_smooth - self._k * width  # smoothed, relative to the current lane's center
        at_speed = speed is not None and speed >= cfg.min_speed_mps
        after_change = t - self._last_lane_change_t < cfg.lane_change_quiet_s
        if at_speed and not after_change:
            self._learn_habit(t, offset)
        habit = self._habit_value
        if habit is None:
            return []  # where we usually drive in the lane isn't known yet
        rel = offset - habit  # from our usual place
        side: Side = "right" if rel > 0 else "left"
        gap = width / 2.0 - abs(rel) - cfg.car_width_m / 2.0
        if gap >= cfg.rearm_gap_m:
            self._armed = {"left": True, "right": True}
            self._drift_side = None
        if v is None or speed is None or not at_speed:
            return []
        toward = v if side == "right" else -v
        if not (gap <= cfg.drift_gap_m and cfg.drift_min_mps <= toward < cfg.lane_change_mps):
            return []
        if not self._armed[side] or t - self._last_alert_t < cfg.cooldown_s:
            return []
        if after_change:
            return []
        window = [(tt, vv) for tt, vv in self._speeds if t - tt <= cfg.drift_window_s]
        if not window or t - window[0][0] < 0.8 * cfg.drift_window_s:
            return []
        if max(abs(vv) for _, vv in window) >= cfg.lane_change_mps:
            return []  # part of a fast sideways move: a lane change
        quick = self._velocity(t, cfg.quick_window_s)
        if quick is not None and abs(quick) >= cfg.lane_change_mps:
            return []  # a lane change starting now (the speed over velocity_window_s still lags behind it)
        sign = 1.0 if side == "right" else -1.0
        steady = sum(1 for _, vv in window if sign * vv > 0.0)
        if steady < cfg.drift_steady_share * len(window):
            return []  # not moving toward the line all along: the start of a sudden move (its speed fit lags)
        past = [pp for tt, pp in self._positions if t - tt >= cfg.drift_window_s]
        if not past:
            return []
        shift = (p - past[-1]) * (1.0 if side == "right" else -1.0)
        if shift < cfg.drift_min_shift_m:
            return []
        self._armed[side] = False
        self._last_alert_t = t
        self._drift_side = side
        return [DriftEvent(t, side, drift_text(side), gap, toward, offset, habit, speed)]

    def _velocity(self, t: float, window_s: float | None = None) -> float | None:
        """Lateral speed (m/s, + = right): a least-squares slope over the last `window_s` (velocity_window_s)."""
        span = self.cfg.velocity_window_s if window_s is None else window_s
        pts = [(tt, pp) for tt, pp in self._positions if t - tt <= span]
        if len(pts) < 5 or pts[-1][0] - pts[0][0] < 0.6 * span:
            return None
        ts = np.array([q[0] for q in pts])
        ps = np.array([q[1] for q in pts])
        ts = ts - ts.mean()
        return float((ts * (ps - ps.mean())).sum() / (ts * ts).sum())

    # ------------------------------------------------------------------------------------------ outputs

    def _vote(self) -> tuple[int | None, int | None, float]:
        """Majority lanes_left / lanes_right over the window (None if not clear), and the confidence."""
        cfg = self.cfg
        out: list[int | None] = []
        shares: list[float] = []
        sides: tuple[list[int | None], list[int | None]] = ([v[1] for v in self._votes], [v[2] for v in self._votes])
        for values in sides:
            answered = [x for x in values if x is not None]
            if len(answered) < cfg.vote_min_frames:
                out.append(None)
                shares.append(0.0)
                continue
            counts: dict[int, int] = {}
            for x in answered:
                counts[x] = counts.get(x, 0) + 1
            value, n = max(counts.items(), key=lambda kv: kv[1])
            share = n / len(answered)
            if share < cfg.vote_share:
                out.append(None)
                shares.append(0.0)
                continue
            out.append(value)
            shares.append(share * len(answered) / max(1, len(values)))
        known = [s for s, o in zip(shares, out, strict=True) if o is not None]
        return out[0], out[1], float(min(known)) if known else 0.0

    def _make_snapshot(self, t: float, m: LaneMeasurement) -> LanesSnapshot:
        lanes_left, lanes_right, confidence = self._vote()
        valid = m.offset_m is not None
        views = tuple(
            LineView(
                line.points,
                round(line.offset_m * self.scale, 3),
                "left" if i == m.left and valid else "right" if i == m.right and valid else "other",
            )
            for i, line in enumerate(m.lines)
        )
        polygon: tuple[Point, ...] = ()
        drift_line = None
        if valid and m.left is not None and m.right is not None:
            polygon = m.lines[m.left].points + tuple(reversed(m.lines[m.right].points))
            if self._drift_side is not None:
                drift_line = m.lines[m.left if self._drift_side == "left" else m.right].points
        lateral = self._speeds[-1][1] if self._speeds and valid else None
        return LanesSnapshot(
            t=t,
            valid=valid,
            lane_polygon=polygon,
            lines=views,
            drift_side=self._drift_side,
            drift_line=drift_line,
            lane_count=None if lanes_left is None or lanes_right is None else lanes_left + 1 + lanes_right,
            lane_index=None if lanes_left is None else lanes_left + 1,
            lanes_left=lanes_left,
            lanes_right=lanes_right,
            confidence=round(confidence, 3),
            offset_m=None if m.offset_m is None else round(m.offset_m, 3),
            lane_width_m=None if m.width_m is None else round(m.width_m, 3),
            lateral_speed_mps=None if lateral is None else round(lateral, 3),
            horizon=round(self._horizon, 4),
            habit_m=None if self._habit_value is None else round(self._habit_value, 3),
        )

    def snapshot(self) -> LanesSnapshot:
        """The latest state (any thread)."""
        with self._lock:
            return self._snapshot

    def context(self) -> list[str]:
        """Lines for the vision model's prompt: what lane detection sees (only what is clear)."""
        snap = self.snapshot()
        if snap.lane_count is not None and snap.lane_index is not None:
            return [
                f"Lane detection: {snap.lane_count} lanes in our direction; "
                f"we are in lane {snap.lane_index} from the left."
            ]
        if snap.lanes_left is not None:
            return [f"Lane detection: {snap.lanes_left} lane(s) to our left (the count to the right is unclear)."]
        if snap.lanes_right is not None:
            return [f"Lane detection: {snap.lanes_right} lane(s) to our right (the count to the left is unclear)."]
        return []


__all__ = [
    "DRIFT_TEXTS",
    "EMPTY",
    "DriftEvent",
    "LaneLine",
    "LaneMeasurement",
    "LaneTracker",
    "LanesSnapshot",
    "LineView",
    "drift_text",
    "vanishing_row",
]
