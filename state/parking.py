"""Parking: where the parked cars are, which spaces between them are open, cars pulling out, whether this looks
like a parking lot, and whether we have parked (V2-M1, "find me a parking spot"; answers/parking.py speaks).

Geometry. A vehicle box's bottom edge touches the ground, so with a pinhole camera at a known height the ground
point follows from the image row: distance Z = f·h / (y − y_horizon), lateral X = (x − cx)·Z / f (+ = right).
For cars parked side by side along an aisle, the bottom edge of each box is the depth of the car's near side, and
consecutive cars' near sides are one stall apart (the "pitch"); the box's inner edge is the far corner of the car's
aisle-side face, which gives the row's aisle-side edge. Every length (car sizes, tolerances, how deep we must see)
is scaled by the measured pitch over its prior, so a moderately wrong focal length (field of view) scales
everything alike and changes no decision; the scale is kept within `scale_min`..`scale_max` (the field of view is
known roughly), and only spacings near the prior pitch measure it (in a sparse row most spacings are gaps). The
horizon row is estimated from the cars (a car of height H whose box is h px tall stands with its bottom
h·camera_height/H px below the horizon), median over many boxes.

Parked cars. `state/vehicle_motion.py` can't tell them while we drive: it cancels our motion only for the
background, not for the cars right beside us. Here a car is parked when it comes closer exactly as fast as we
drive: our speed is the GPS speed when there is a fix, else the cars' common approach rate (in a lot most cars are
parked: the median of −dZ/dt over the near cars beside the aisle). Comparisons pause while we turn (positions sweep
sideways). A box cut off by the frame's bottom edge (our own hood, a car right beside us) gives no position, nor
one cut off at the side once its nearest corner has left the frame: its ground contact is out of view.

Spaces. Between two parked neighbours a spacing of ≥ `gap_ratio` pitches leaves a space. It is a candidate while
its aisle-side face (where a car parked there would show its front) is in view, not hidden by other vehicles, and
nothing tracked stands in it; after `confirm_s` of that while we approach, a "candidate" event lets the vision
model start its check early. It becomes a "spot" once we can also see `min_visible_depth_m` into the middle of
the stall past the nearer neighbour: a small car pulled in behind a big one stays hidden until then, and is seen
(and resets the candidate) by the time we can look that deep. The reach grows as we approach.

Also: "pulling_out" (a car parked for a while starts to move), "lot" (enough parked cars in rows while we crawl:
this looks like a parking lot), "parked" (we stopped in a space: stopped after a sharp turn, or for a long time).
Modes: "lot" (cars side by side, ~2.7 m stalls) or "street" (parallel parking along a curb, ~6 m per car).
Called once per frame on the perception thread; not thread-safe on its own (WorldState serializes it).
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from core.config import ParkingConfig
from perception.detector import VEHICLE_CLASSES, Track

Box = tuple[float, float, float, float]
OBSTACLE_CLASSES = VEHICLE_CLASSES | {"person", "bicycle"}  # anything standing in a space fills it
SIDES = ("left", "right")


@dataclass(frozen=True, slots=True)
class Camera:
    """Pinhole camera looking level along the road; image rows grow downwards."""

    f: float  # focal length (px)
    cx: float
    height_m: float

    @classmethod
    def from_fov(cls, width: int, hfov_deg: float, height_m: float) -> Camera:
        """From the image width and horizontal field of view."""
        return cls(width / 2 / math.tan(math.radians(hfov_deg) / 2), width / 2, height_m)

    def ground(self, x: float, y: float, horizon: float) -> tuple[float, float] | None:
        """(X lateral, Z ahead) in m of an image point on the ground, or None at or above the horizon."""
        dy = y - horizon
        if dy <= 1.0:
            return None
        z = self.f * self.height_m / dy
        return (x - self.cx) * z / self.f, z

    def project(self, x: float, z: float, up: float, horizon: float) -> tuple[float, float]:
        """Image point of the world point (x right, z ahead, `up` m above the ground)."""
        return self.cx + self.f * x / z, horizon + self.f * (self.height_m - up) / z


@dataclass(frozen=True, slots=True)
class ParkedCar:
    """A parked vehicle in the row beside our path."""

    track_id: int
    cls: str
    side: str  # "left" | "right"
    x: float  # lateral offset of its aisle-side edge (m, + = right)
    z: float  # distance ahead of its near side (m)
    box: Box
    parked_since: float


@dataclass(frozen=True, slots=True)
class Space:
    """An open space (one or more stalls) between two parked cars."""

    key: tuple[str, int, int]  # (side, near car id, far car id)
    side: str
    near: ParkedCar  # the car right before the space (we pass it first)
    far: ParkedCar
    stalls: int
    z: float  # distance ahead of the space's near edge (m)
    region: Box  # image box of the space's aisle-side face (for the model check and overlays)
    cars_before: int  # parked cars on that side between us and the space
    pitch: float
    seen_since: float
    visible_depth: float  # how deep into the stall we can see now (m)
    confirmed: bool  # seen deep enough into: a "spot"
    along: float  # our travelled distance + z: where the space is along our path (dedupes re-detections)


@dataclass(frozen=True, slots=True)
class ParkingEvent:
    """Something the parking watch may act on."""

    kind: str  # "candidate" | "spot" | "taken" | "pulling_out" | "lot" | "parked"
    t: float
    space: Space | None = None
    car: ParkedCar | None = None


@dataclass(frozen=True, slots=True)
class ParkingSnapshot:
    """Read-only parking state at one moment."""

    t: float
    mode: str
    horizon: float
    speed: float | None  # our speed (m/s, camera scale) from the parked cars, None if unknown
    rows: dict[str, tuple[ParkedCar, ...]]
    pitch: dict[str, float]
    candidates: tuple[Space, ...]  # spaces that look open now (spots and not-yet-confirmed ones)
    lot_like: bool
    stopped_s: float  # how long we have been standing still (0 while moving)
    along: float = 0.0  # how far we have driven (m, camera scale): spaces behind us have a smaller `along`


EMPTY = ParkingSnapshot(0.0, "lot", 0.0, None, {"left": (), "right": ()}, {}, (), False, 0.0)


@dataclass(slots=True)
class _Car:
    """Per-track history (ground positions) and parked/moving state."""

    cls: str
    box: Box = (0.0, 0.0, 0.0, 0.0)  # latest box
    samples: deque[tuple[float, float, float]] = field(default_factory=deque)  # (t, x, z)
    parked_since: float | None = None
    moving_since: float | None = None
    was_parked_s: float = 0.0  # how long it had been parked when it last started to move
    reported_pull_out: bool = False
    parked_x: float | None = None  # lateral position while last parked...
    parked_yaw: float = 0.0  # ...and our accumulated heading then (degrees)
    in_row: bool = False  # was seen parked in a row beside the aisle
    last_seen: float = 0.0


@dataclass(slots=True)
class _Candidate:
    first_seen: float
    first_along: float
    last_valid: float
    near: ParkedCar  # the nearer neighbour when last seen...
    near_along: float  # ...and how far we had driven then (it is predicted once it leaves the frame)
    index: int  # its position in the row (cars before the space)
    deep_frames: int = 0  # frames in which we could see `min_visible_depth_m` into it (and it was empty)
    announced_candidate: bool = False
    confirmed: bool = False
    space: Space | None = None  # as last seen open


def _fit_slope(ts: np.ndarray, vs: np.ndarray) -> float:
    """Least-squares slope of vs over ts."""
    tc = ts - ts.mean()
    denom = float((tc * tc).sum())
    return 0.0 if denom <= 0 else float((tc * (vs - vs.mean())).sum() / denom)


def coverage(region: Box, covers: Iterable[Box], grid: int = 16) -> float:
    """Fraction of `region` covered by the union of `covers` (sampled on a grid)."""
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        return 1.0
    xs = np.linspace(x1, x2, grid)
    ys = np.linspace(y1, y2, grid)
    gx, gy = np.meshgrid(xs, ys)
    hidden = np.zeros_like(gx, dtype=bool)
    for bx1, by1, bx2, by2 in covers:
        hidden |= (gx >= bx1) & (gx <= bx2) & (gy >= by1) & (gy <= by2)
    return float(hidden.mean())


class ParkingTracker:
    """Per-frame parking state; `update` returns the events of this frame."""

    def __init__(self, cfg: ParkingConfig, hfov_deg: float) -> None:
        self.cfg = cfg
        self.hfov_deg = hfov_deg
        self.mode = "lot"
        self.reset()

    def reset(self) -> None:
        """Forget everything (new video segment)."""
        self._cars: dict[int, _Car] = {}
        self._candidates: dict[tuple[str, int, int], _Candidate] = {}
        self._horizon_samples: deque[float] = deque(maxlen=400)
        self._spacings: dict[str, deque[tuple[float, float]]] = {side: deque() for side in SIDES}
        self._row_ids: deque[tuple[float, int]] = deque()  # (t, track id) of cars seen parked in a row
        self._lot_like = False
        self._lot_last_match: float | None = None
        self._scale = 1.0  # measured pitch / prior pitch: the camera's metric scale (1.0 if the FOV is right)
        self._along = 0.0  # how far we have driven (m, camera scale), from our estimated speed
        self._yaw: deque[tuple[float, float]] = deque()  # (t, accumulated yaw in degrees)
        self._yaw_total = 0.0
        self._stopped_since: float | None = None
        self._parked_reported = False
        self._last_t: float | None = None
        self._snapshot = EMPTY

    def set_mode(self, mode: str) -> None:
        """ "lot" (cars side by side) or "street" (parallel parking along a curb); spacings restart."""
        if mode not in ("lot", "street"):
            raise ValueError(f"unknown parking mode: {mode}")
        if mode != self.mode:
            self.mode = mode
            self._candidates.clear()
            for spacings in self._spacings.values():
                spacings.clear()
            self._scale = 1.0

    # ----------------------------------------------------------------------------------- sizes

    @property
    def _prior_pitch(self) -> float:
        return self.cfg.lot_pitch_m if self.mode == "lot" else self.cfg.curb_pitch_m

    @property
    def _across(self) -> float:
        """A parked car's extent along the row (its width side by side; its length along a curb), scaled."""
        cfg = self.cfg
        return (cfg.car_width_m if self.mode == "lot" else cfg.car_length_m) * self._scale

    @property
    def _deep(self) -> float:
        """A parked car's extent into the row, scaled."""
        cfg = self.cfg
        return (cfg.car_length_m if self.mode == "lot" else cfg.car_width_m) * self._scale

    def _m(self, metres: float) -> float:
        """A length in the camera's (possibly mis-scaled) metres."""
        return metres * self._scale

    # ------------------------------------------------------------------------------------ update

    def update(
        self,
        t: float,
        width: int,
        height: int,
        tracks: Sequence[Track],
        ego_stopped: bool,
        homography: np.ndarray | None,
        gps_speed: float | None = None,
    ) -> list[ParkingEvent]:
        """Fold one frame in (`gps_speed`: our GPS speed in m/s if there's a fresh fix); returns its events."""
        cfg = self.cfg
        cam = Camera.from_fov(width, self.hfov_deg, cfg.camera_height_m)
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        events: list[ParkingEvent] = []

        vehicles = [tr for tr in tracks if tr.cls in VEHICLE_CLASSES]
        horizon = self._horizon([tr for tr in vehicles if not self._cut_off(tr.box, width, height)], height)
        yaw_rate = self._update_yaw(t, dt, homography, cam, horizon)
        turning = yaw_rate is not None and abs(yaw_rate) > cfg.max_yaw_rate_dps
        self._positions(t, vehicles, cam, horizon, turning, width, height)

        speed = self._our_speed(t, ego_stopped, turning, gps_speed)
        if speed is not None:
            self._along += speed * dt
        self._update_parked_states(t, speed, cam, events)
        self._drop_stale(t)

        rows = {side: self._row(side, t, vehicles) for side in SIDES}
        for side, row in rows.items():
            for near, far in itertools.pairwise(row):
                self._spacings[side].append((t, far.z - near.z))
            while self._spacings[side] and t - self._spacings[side][0][0] > cfg.pitch_window_s:
                self._spacings[side].popleft()
            for car in row:
                self._row_ids.append((t, car.track_id))
                self._cars[car.track_id].in_row = True
        while self._row_ids and t - self._row_ids[0][0] > cfg.lot_window_s:
            self._row_ids.popleft()
        self._update_scale(t)

        candidates = [] if turning else self._spaces(t, rows, tracks, cam, horizon, speed, events)
        self._lot(t, speed, events)
        stopped_s = self._parked(t, ego_stopped, events)
        self._snapshot = ParkingSnapshot(
            t=t,
            mode=self.mode,
            horizon=horizon,
            speed=speed,
            rows={side: tuple(row) for side, row in rows.items()},
            pitch={side: self._pitch(side) for side in SIDES},
            candidates=tuple(candidates),
            lot_like=self._lot_like,
            stopped_s=stopped_s,
            along=self._along,
        )
        return events

    def snapshot(self) -> ParkingSnapshot:
        """The latest state."""
        return self._snapshot

    # ----------------------------------------------------------------------------------- pieces

    def _cut_off(self, box: Box, width: int, height: int) -> bool:
        """The box touches the frame's bottom or side edge."""
        margin = self.cfg.edge_margin_px
        x1, _, x2, y2 = box
        return y2 >= height - margin or x1 <= margin or x2 >= width - margin

    def _ground_visible(self, box: Box, width: int, height: int, cam: Camera, horizon: float) -> bool:
        """The box's bottom edge is the car's ground contact: not cut off by the frame's bottom, and, if cut off
        at the side, its nearest corner (in line with its inner edge, one car "across" nearer) is still in view."""
        margin = self.cfg.edge_margin_px
        x1, _, x2, y2 = box
        if y2 >= height - margin:
            return False
        left, right = x1 <= margin, x2 >= width - margin
        if not left and not right:
            return True
        point = cam.ground(x2 if left else x1, y2, horizon)
        if point is None or (left and right):
            return False
        z = point[1]
        corner = cam.cx + ((x2 if left else x1) - cam.cx) * (z + self._across) / z
        return margin < corner < width - margin

    def _horizon(self, vehicles: Sequence[Track], height: int) -> float:
        cfg = self.cfg
        for tr in vehicles:
            if tr.cls == "car" and tr.height > 8:
                self._horizon_samples.append(tr.box[3] - tr.height * cfg.camera_height_m / cfg.car_height_m)
        if len(self._horizon_samples) < cfg.horizon_samples:
            return cfg.horizon_prior * height
        estimate = float(np.median(self._horizon_samples))
        return float(np.clip(estimate, 0.25 * height, 0.7 * height))

    def _update_yaw(
        self, t: float, dt: float, homography: np.ndarray | None, cam: Camera, horizon: float
    ) -> float | None:
        """Accumulate our heading change from the frame-to-frame homography; returns the yaw rate (°/s)."""
        rate = None
        if homography is not None and dt > 0:
            p = homography @ np.array([cam.cx, horizon, 1.0])
            if abs(p[2]) > 1e-9:
                dx = p[0] / p[2] - cam.cx  # the point at the horizon straight ahead moves sideways as we turn
                d_yaw = math.degrees(math.atan2(dx, cam.f))
                self._yaw_total += d_yaw
                rate = d_yaw / dt
        self._yaw.append((t, self._yaw_total))
        while self._yaw and t - self._yaw[0][0] > self.cfg.parked_turn_window_s + self.cfg.parked_after_s:
            self._yaw.popleft()
        return rate

    def _positions(
        self,
        t: float,
        vehicles: Sequence[Track],
        cam: Camera,
        horizon: float,
        turning: bool,
        width: int,
        height: int,
    ) -> None:
        """Ground position of every vehicle whose ground contact is in view: the bottom edge is its near side (Z);
        for a car beside our path the inner edge is the far corner of its aisle-side face (one car "across" deeper),
        which gives the row's edge."""
        lane = self._m(self.cfg.lane_half_width_m)
        for tr in vehicles:
            car = self._cars.setdefault(tr.track_id, _Car(tr.cls))
            car.cls, car.box, car.last_seen = tr.cls, tr.box, t
            if turning:
                car.samples.clear()  # positions from before a turn can't be compared with those after it
                continue
            if not self._ground_visible(tr.box, width, height, cam, horizon):
                continue  # cut off: no position this frame (its history ages out)
            x1, _, x2, y2 = tr.box
            inner = x1 if (x1 + x2) / 2 > cam.cx else x2
            point = cam.ground(inner, y2, horizon)
            if point is not None and abs(point[0]) >= lane:
                point = (point[0] * (point[1] + self._across) / point[1], point[1])
            if point is None or turning:
                car.samples.clear()  # positions from before a turn or above the horizon can't be compared
                continue
            car.samples.append((t, point[0], point[1]))
            while car.samples and t - car.samples[0][0] > self.cfg.history_s:
                car.samples.popleft()

    def _velocity(self, car: _Car, t: float) -> tuple[float, float] | None:
        """(dX/dt, dZ/dt) over the last `fit_s`, or None without enough history."""
        fit = self._fit(car, t)
        return None if fit is None else (fit[0], fit[1])

    def _fit(self, car: _Car, t: float) -> tuple[float, float, int, float] | None:
        """(dX/dt, dZ/dt, samples, time span) over the last `fit_s`, or None without enough history."""
        recent = [s for s in car.samples if t - s[0] <= self.cfg.fit_s]
        if len(recent) < 5 or recent[-1][0] - recent[0][0] < 0.5 * self.cfg.fit_s:
            return None
        arr = np.array(recent)
        span = float(arr[-1, 0] - arr[0, 0])
        return _fit_slope(arr[:, 0], arr[:, 1]), _fit_slope(arr[:, 0], arr[:, 2]), len(recent), span

    def _our_speed(self, t: float, ego_stopped: bool, turning: bool, gps_speed: float | None) -> float | None:
        """Our speed (camera-scale m/s): from GPS if there's a fix, else the median rate at which the near cars
        beside our path come closer (most of them are parked)."""
        if turning:
            return None
        if gps_speed is not None:
            return max(0.0, gps_speed) * self._scale
        if ego_stopped:
            return 0.0
        lane, near = self._m(self.cfg.lane_half_width_m), self._m(self.cfg.speed_max_z_m)
        rates = []
        for car in self._cars.values():
            if car.last_seen != t or not car.samples or car.samples[-1][0] != t:
                continue
            if abs(car.samples[-1][1]) < lane or car.samples[-1][2] > near:
                continue
            v = self._velocity(car, t)
            if v is not None:
                rates.append(-v[1])
        if len(rates) >= 3:
            return max(0.0, float(np.median(rates)))
        return None

    def _update_parked_states(self, t: float, speed: float | None, cam: Camera, events: list[ParkingEvent]) -> None:
        cfg = self.cfg
        if speed is None:
            return  # our own motion is unknown: keep every car's state
        pitch = self._m(self._prior_pitch)
        base = cfg.static_rel_tol * speed + cfg.static_abs_tol * pitch
        base_x = cfg.static_abs_tol_x * pitch
        for track_id, car in self._cars.items():
            if car.last_seen != t:
                continue
            fit = self._fit(car, t)
            if fit is None:
                continue
            vx, vz, n, span = fit
            _, x, z = car.samples[-1]
            # Speed noise of a slope fitted to n samples over `span` s, from box jitter: depth from one image row
            # (σz = z²/(f·h)·σy) grows fast with distance; lateral (σx = z/f·σx_px) slowly.
            k = math.sqrt(12.0 / (n * span * span))
            noise_z = 3.0 * k * z * z / (cam.f * cam.height_m) * cfg.box_jitter_px
            noise_x = 3.0 * k * z / cam.f * cfg.box_jitter_px
            if abs(vz + speed) < base + noise_z and abs(vx) < base_x + noise_x:
                car.moving_since = None
                car.parked_x, car.parked_yaw = x, self._yaw_total
                if car.parked_since is None:
                    car.parked_since = t
                continue
            if car.moving_since is None:
                car.moving_since = t
                car.was_parked_s = 0.0 if car.parked_since is None else t - car.parked_since
            if t - car.moving_since < cfg.move_confirm_s:
                continue  # a blip (our speed estimate lags when we brake): still parked
            car.parked_since = None
            # Pulling out of a stall (or away from the curb) is a move from the row toward our path; a car moving
            # off ahead of us in the aisle, or depths drifting together far away, is not. Our own slow turn since
            # then shifts every car sideways (by z·tan(yaw)): taken out.
            x_own = x - z * math.tan(math.radians(self._yaw_total - car.parked_yaw))
            toward = 0.0 if car.parked_x is None else abs(car.parked_x) - abs(x_own)
            if (
                not car.reported_pull_out
                and car.in_row
                and car.was_parked_s >= cfg.pull_out_parked_s
                and t - car.moving_since >= cfg.pull_out_moving_s
                and car.parked_x is not None
                and car.parked_x * x_own > 0
                and toward >= self._m(cfg.pull_out_min_m)
                and z <= self._m(cfg.pull_out_max_z_m)
            ):
                car.reported_pull_out = True
                side = "right" if x > 0 else "left"
                events.append(ParkingEvent("pulling_out", t, car=ParkedCar(track_id, car.cls, side, x, z, car.box, t)))

    def _drop_stale(self, t: float) -> None:
        stale = [k for k, car in self._cars.items() if t - car.last_seen > self.cfg.history_s]
        for k in stale:
            del self._cars[k]

    def _row(self, side: str, t: float, vehicles: Sequence[Track]) -> list[ParkedCar]:
        """The parked cars of the row next to our path on one side, near to far."""
        cfg = self.cfg
        sign = 1.0 if side == "right" else -1.0
        lane, row_max = self._m(cfg.lane_half_width_m), self._m(cfg.row_max_x_m)
        parked = []
        for tr in vehicles:
            car = self._cars.get(tr.track_id)
            if car is None or not car.samples or car.samples[-1][0] != t or car.parked_since is None:
                continue
            if t - car.parked_since < cfg.min_parked_s:
                continue
            _, x, z = car.samples[-1]
            if sign * x < lane or sign * x > row_max or z <= self._m(0.5):
                continue
            parked.append(ParkedCar(tr.track_id, tr.cls, side, x, z, tr.box, car.parked_since))
        if not parked:
            return []
        nearest = min(sign * c.x for c in parked)
        return sorted((c for c in parked if sign * c.x <= nearest + self._m(cfg.row_depth_m)), key=lambda c: c.z)

    def _measured_pitch(self, side: str) -> float | None:
        """The median of recent neighbour spacings near the prior pitch (gaps and missed cars are larger, split
        boxes smaller), or None with too few of them. Known limit: the window also holds one-stall gaps at a high
        scale, so a row where most spacings are single gaps can measure the gap (tests/test_v2_review.py, xfail);
        tighter windows, lower quartiles and harmonic voting all did worse on the real lot clips."""
        cfg = self.cfg
        prior = self._prior_pitch
        lo, hi = cfg.pitch_ratio_min * prior * cfg.scale_min, cfg.pitch_ratio_max * prior * cfg.scale_max
        spacings = [s for _, s in self._spacings[side] if lo <= s <= hi]
        return float(np.median(spacings)) if len(spacings) >= cfg.pitch_min_samples else None

    def _pitch(self, side: str) -> float:
        """The row's pitch (camera-scale m): measured on that side, else the prior at the current scale."""
        measured = self._measured_pitch(side)
        return self._m(self._prior_pitch) if measured is None else measured

    def _update_scale(self, t: float) -> None:
        measured = [p for p in (self._measured_pitch(side) for side in SIDES) if p is not None]
        if measured:
            ratio = float(np.mean(measured)) / self._prior_pitch
            self._scale = float(np.clip(ratio, self.cfg.scale_min, self.cfg.scale_max))

    def _spaces(
        self,
        t: float,
        rows: dict[str, list[ParkedCar]],
        tracks: Sequence[Track],
        cam: Camera,
        horizon: float,
        speed: float | None,
        events: list[ParkingEvent],
    ) -> list[Space]:
        cfg = self.cfg
        out: list[Space] = []
        seen: set[tuple[str, int, int]] = set()
        for side, row in rows.items():
            pitch = self._pitch(side)
            pairs: list[tuple[int, ParkedCar, ParkedCar]] = []
            for index, (near, far) in enumerate(itertools.pairwise(row)):
                if far.z - near.z >= cfg.gap_ratio * pitch and far.z <= cfg.max_ahead_pitches * pitch:
                    pairs.append((index, near, far))
            # A candidate whose nearer neighbour has left the frame (we're close to the space now; a narrow camera
            # loses the neighbour before it can see into the stall): predict the neighbour from our own motion.
            row_ids = {c.track_id: c for c in row}
            paired = {(n.track_id, f.track_id) for _, n, f in pairs}
            for (cand_side, near_id, far_id), old in self._candidates.items():
                if cand_side != side or near_id in row_ids or far_id not in row_ids or (near_id, far_id) in paired:
                    continue
                z = old.near.z - (self._along - old.near_along)
                if z > self._m(0.5):
                    near = ParkedCar(
                        near_id, old.near.cls, side, old.near.x, z, (0.0, 0.0, 0.0, 0.0), old.near.parked_since
                    )
                    pairs.append((old.index, near, row_ids[far_id]))
            for index, near, far in pairs:
                spacing = far.z - near.z
                key = (side, near.track_id, far.track_id)
                region = self._space_region(near, pitch, cam, horizon)
                status = self._open_front(region, near, far, tracks, cam, horizon)
                if status != "open":
                    gone = self._candidates.pop(key, None)  # seen but not open now: the streak starts over
                    if status == "blocked" and gone is not None and gone.confirmed and gone.space is not None:
                        events.append(ParkingEvent("taken", t, space=gone.space))  # something is in it now
                    continue
                seen.add(key)
                cand = self._candidates.get(key)
                if cand is None or t - cand.last_valid > cfg.grace_s:
                    cand = self._candidates[key] = _Candidate(t, self._along, t, near, self._along, index)
                cand.last_valid = t
                if near.box != (0.0, 0.0, 0.0, 0.0):  # (a real detection, not a prediction)
                    cand.near, cand.near_along, cand.index = near, self._along, index
                depth = self._visible_depth(near, far, pitch)
                deep = depth >= self._m(cfg.min_visible_depth_m)
                cand.deep_frames += int(deep)
                approached = self._along - cand.first_along
                still = speed is not None and speed < 0.2 * pitch
                looked = t - cand.first_seen >= cfg.confirm_s and (
                    approached >= cfg.confirm_approach_pitches * pitch
                    or (still and t - cand.first_seen >= cfg.confirm_stopped_s)
                )
                # Open-looking for a while from several viewpoints, and now seen deep enough into to be sure.
                ready = looked and deep and cand.deep_frames >= 2
                space = Space(
                    key=key,
                    side=side,
                    near=near,
                    far=far,
                    stalls=max(1, round(spacing / pitch) - 1),
                    z=near.z + self._across,
                    region=region,
                    cars_before=cand.index + 1,
                    pitch=pitch,
                    seen_since=cand.first_seen,
                    visible_depth=depth,
                    confirmed=cand.confirmed or ready,
                    along=self._along + near.z + self._across,
                )
                if looked and not cand.announced_candidate:
                    cand.announced_candidate = True
                    events.append(ParkingEvent("candidate", t, space=space))
                if ready and not cand.confirmed:
                    cand.confirmed = True
                    events.append(ParkingEvent("spot", t, space=space))
                cand.space = space
                out.append(space)
            # A confirmed space whose neighbours are both still in view but no longer next to each other: a car
            # has parked in it.
            for key, cand in list(self._candidates.items()):
                _, near_id, far_id = key
                if key[0] != side or key in seen or not cand.confirmed or cand.space is None:
                    continue
                if near_id in row_ids and far_id in row_ids:
                    between = [c for c in row if row_ids[near_id].z < c.z < row_ids[far_id].z]
                    if between:
                        del self._candidates[key]
                        events.append(ParkingEvent("taken", t, space=cand.space))
        for key in [k for k, c in self._candidates.items() if k not in seen and t - c.last_valid > cfg.grace_s]:
            del self._candidates[key]
        return out

    def _space_region(self, near: ParkedCar, pitch: float, cam: Camera, horizon: float) -> Box:
        """Image box of the aisle-side face of the first free stall after `near`: where a car parked there would
        show its front (that face stays in view past the neighbours; the rest of the stall is hidden behind them)."""
        sign = 1.0 if near.side == "right" else -1.0
        z0 = near.z + pitch
        z1 = z0 + self._across
        x0 = near.x
        x1 = near.x + sign * self._m(self.cfg.space_front_m)
        height = self.cfg.car_height_m * self._scale
        points = [cam.project(x, z, up, horizon) for x in (x0, x1) for z in (z0, z1) for up in (0.0, height)]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _visible_depth(self, near: ParkedCar, far: ParkedCar, pitch: float) -> float:
        """How deep into the middle of the first free stall (m, from the row's aisle-side edge) we can see past the
        nearer neighbour: sight lines pass its far aisle-side corner (|x|, near.z + across)."""
        middle = min(far.z, near.z + pitch + self._across / 2)
        return abs(near.x) * (middle / (near.z + self._across) - 1.0)

    def _open_front(
        self, region: Box, near: ParkedCar, far: ParkedCar, tracks: Sequence[Track], cam: Camera, horizon: float
    ) -> str:
        """ "open" if the space's aisle-side face is in the frame and not hidden by other vehicles, and nothing
        tracked stands anywhere in the space; else "unseen" (out of the frame, hidden) or "blocked" (filled)."""
        cfg = self.cfg
        rx1, _, rx2, ry2 = region
        inside = (min(rx2, 2 * cam.cx) - max(rx1, 0.0)) / max(rx2 - rx1, 1e-6)
        if inside < 0.6:
            return "unseen"  # mostly out of the frame
        others = [
            tr.box
            for tr in tracks
            if tr.cls in VEHICLE_CLASSES and tr.box[3] > ry2 + 2 and tr.track_id not in (near.track_id, far.track_id)
        ]
        # The near neighbour hides only the part of the face behind its own aisle-side corner.
        if coverage(region, [*others, near.box] if near.box[2] > near.box[0] else others) > cfg.max_occlusion:
            return "unseen"
        sign = 1.0 if near.side == "right" else -1.0
        z_from, z_to = near.z + self._across * 0.5, far.z - self._m(0.2)
        for tr in tracks:
            if tr.cls not in OBSTACLE_CLASSES or tr.track_id in (near.track_id, far.track_id):
                continue
            x1, _, x2, y2 = tr.box
            point = cam.ground((x1 + x2) / 2, y2, horizon)
            if point is None:
                continue
            depth = sign * (point[0] - near.x)  # how far into the row, from its aisle-side edge
            if z_from <= point[1] <= z_to and -self._m(0.5) <= depth <= self._deep:
                return "blocked"  # someone or something stands in the space
        return "open"

    def _lot(self, t: float, speed: float | None, events: list[ParkingEvent]) -> None:
        """Crawling past enough parked cars in rows: a lot (held for `lot_hold_s` after it stops matching)."""
        cfg = self.cfg
        slow = speed is not None and speed <= self._m(cfg.lot_max_speed_mps)
        if slow and len({track_id for _, track_id in self._row_ids}) >= cfg.lot_min_parked:
            self._lot_last_match = t
        lot_like = self._lot_last_match is not None and t - self._lot_last_match <= cfg.lot_hold_s
        if lot_like and not self._lot_like:
            events.append(ParkingEvent("lot", t))
        self._lot_like = lot_like

    def _parked(self, t: float, ego_stopped: bool, events: list[ParkingEvent]) -> float:
        """How long we've been stopped; reports "parked" once per stop that looks like parking."""
        cfg = self.cfg
        if not ego_stopped:
            self._stopped_since = None
            self._parked_reported = False
            return 0.0
        if self._stopped_since is None:
            self._stopped_since = t
        stopped_s = t - self._stopped_since
        if self._parked_reported or stopped_s < cfg.parked_after_s:
            return stopped_s
        since = self._stopped_since
        window = [yaw for when, yaw in self._yaw if since - cfg.parked_turn_window_s <= when <= since]
        turned = (max(window) - min(window)) if window else 0.0
        if turned >= cfg.parked_turn_deg or stopped_s >= cfg.parked_long_stop_s:
            self._parked_reported = True
            events.append(ParkingEvent("parked", t))
        return stopped_s
