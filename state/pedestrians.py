"""Pedestrians and cyclists in our path, or about to step into it (V2-M3; answers/alerts.py speaks the warning).

Who. YOLO's "person" and "bicycle" tracks. A person whose box overlaps a bicycle's is a cyclist (their union box
reaches down to the wheels); one on a motorcycle is a rider (a vehicle: ignored), and one mostly inside a vehicle's
box but above its bottom edge is in it or hidden behind it (bus passengers, drivers: ignored). Someone on a bicycle
for `rider_frames` is its rider: for `rider_memory_s` after (a dropout of the person's detection) the bicycle stands
in for their track (one alert each). Any other bicycle is parked (at a rack, a pole, the curb; someone who stood
behind it is no rider): tracked, never warned about.

Where. A pinhole camera at a known height (state/parking.py's `Camera`) places each box on the ground (x right,
z ahead, metres). Two distance cues: the bottom edge touches the ground, so with the horizon row (estimated from car
boxes by the parking tracker) z = f·h_cam / (y_bottom − horizon); and a standing person (or rider on a bicycle) is
~1.7 m tall, so z = f·H / box_height. The first breaks on hills (the SF dev clip's crest: 13 m for someone 20 m
away), when braking pitches the camera and when the horizon estimate is off (the error shrinks as people come
closer, so they seem to drift sideways); the second for children and cut-off boxes. They are combined in log space
(`height_weight`), each alone when the other can't be measured. x is measured from where we're heading: dashcams
are often aimed a few degrees off the car's axis (one lot clip ~6°: "straight ahead" in the image was the sidewalk
15 m out), so the image column we head towards is the median focus of expansion of the frame-to-frame homography
while we drive (`aim_*`; the fitted homography damps the ground's expansion, so it gives the direction, not our
speed). Not placed: a box cut off at a side edge and at the top or bottom (the windshield
mount, a reflection, someone right beside our door).

The path zone (`PathZone`): `half_width`
either side of a centreline straight ahead, out to `min_length` + `headway_s` × our speed (GPS, else
`default_speed_mps` while moving). Someone moving away with us (a cyclist riding ahead at our speed) is closed in on
more slowly: their measured approach (+ `closing_margin_mps` for noise) shortens it, never lengthens it, and until it
is measured (a new track, ~0.3 s) it counts as that slow: beyond `min_length` + `headway_s` × the margin (9 m) a new
track waits for its motion to be known; nearer, anything in our path is warned about at once. It bends with
our turn rate (the homography's yaw, read at the focus of expansion) over our speed, less `curvature_uncertainty`
(on straight dev-clip roads the reading is off by up to ~1.5°/s at 10 m/s, growing with speed), and narrows far
out by ½ · that · z² (never below `min_half_width_m`: what is right in front of our car stays in it), so what an
unknown bend may have taken off our path isn't called in it. Turning faster than `turn_dps` is a real turn: no
predictions, a shorter path. Without GPS the turn radius assumes `default_speed_mps` (the homography is no
speedometer), so slow turns bend too little.

About to enter. Outside the zone, within `predict_max_z_m` (or `predict_max_s` at our speed, if further), moving
towards it fast enough to be inside within
`predict_s` at the lateral speed in excess of what static things seem to do: unmodelled gentle curves (z ×
`yaw_noise_dps`) and the distance drifting (|x| × `depth_drift`). A box that jumps (an occlusion ending, a
re-detection: `jump_rel`) restarts its history, so no speed is measured across the jump. No prediction while we turn,
or for a box cut off at a frame edge (its visible centre moves as it comes into view).

Never warned about: a box cut off at a side edge (beside us, its centre unknown), a parked bicycle, and, beyond
`tall_min_z_m` (nearer, our hood can hide feet), a "person" much taller than one standing where its box touches
the ground (`max_tall_ratio`: a tree or pole taken for a person, or someone whose legs are hidden behind a car),
even with the horizon `tall_horizon_slack_deg` higher (a horizon estimated too low makes real people look tall).

When. Only while we move: not stopped by ego motion and, when there is a GPS fix, not slower than `min_speed_mps`
(people crossing in front of us while we wait at a red light are normal); when ego motion can't be measured (it
holds its last state), GPS at `gps_moving_mps` or more means moving. Someone in the zone when we start moving
is warned about then. In the zone (or entering) for `confirm_frames` observations in a row, so a one-frame false
detection says nothing. Once per track; a new track of the same kind appearing where a warned one was just lost
(an ID switch) inherits its warning.

On the road. With the lanes model running (V2-M5), YOLOPv2's drivable area of a recent frame (`DrivableArea`, at
most `road_max_age_s` old) also has a say: someone counts as in the path only if at least `road_min_share` of the
ground beside their feet is road (strips left and right of the box: `DrivableArea.road_beside`), and only at least
`road_gate_min_offset_m` from the path's centreline, so a man walking along the sidewalk next to our corridor isn't
"ahead" while someone right in front of us always is. Someone on the sidewalk moving into the path is still "entering":
stepping off the curb is exactly what that warning is for. Without a mask (lanes off, a stale one) the zone alone
decides.

Called once per frame on the world thread (WorldState); not thread-safe on its own.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from core.config import PedestriansConfig
from perception.detector import VEHICLE_CLASSES, Track
from state.parking import Camera

Box = tuple[float, float, float, float]
PEDESTRIAN, CYCLIST, BICYCLE = "pedestrian", "cyclist", "bicycle"  # BICYCLE: never seen with a rider (parked)
IN_PATH, ENTERING, CROSSING = "in_path", "entering", "crossing"


@dataclass(frozen=True, slots=True)
class PathZone:
    """Our path ahead on the ground (x right, z ahead, metres): within `half_width` of a centreline that bends by
    `curvature` (1/m, + = to the right), narrowing far out by how uncertain that bend is, out to a depth that grows
    with how fast we close in on something."""

    half_width: float
    min_length: float
    headway_s: float
    max_length: float
    speed: float  # our speed (m/s)
    curvature: float = 0.0
    # rad/s (+ = right) the path bends with (the measured turn less its uncertainty): static things sweep sideways at
    # about −turn_rate·z; the rest of a real turn (up to the uncertainty) is left to the noise floors
    turn_rate: float = 0.0
    uncertainty: float = 0.0  # how far off the curvature can be (1/m)
    min_half_width: float = 1.0

    def center(self, z: float) -> float:
        """Lateral position of the centreline at depth z."""
        return 0.5 * self.curvature * z * z

    def half_width_at(self, z: float) -> float:
        """Half the path's width at depth z: less far out, where an unknown bend could have taken it."""
        return max(self.min_half_width, self.half_width - 0.5 * self.uncertainty * z * z)

    def lateral_speed(self, vx: float, z: float) -> float:
        """A measured sideways speed at depth z with our own turning taken out (+ = right)."""
        return vx + self.turn_rate * z

    def offset(self, x: float, z: float) -> float:
        """Signed lateral offset of a ground point from the centreline (m, + = right)."""
        return x - self.center(z)

    def length(self, closing: float | None = None) -> float:
        """How far ahead the path reaches for something we close in on at `closing` m/s (None: our speed)."""
        rate = self.speed if closing is None else closing
        return min(self.max_length, self.min_length + max(0.0, rate) * self.headway_s)

    def contains(self, x: float, z: float, closing: float | None = None) -> bool:
        """The ground point is in the path."""
        return 0.0 < z <= self.length(closing) and abs(self.offset(x, z)) <= self.half_width_at(z)


@dataclass(frozen=True, slots=True)
class DrivableArea:
    """YOLOPv2's drivable area of one frame (perception.lanes.LaneMasks.drivable): 1 = road, stretched over the
    whole frame (mask pixel (r, c) covers frame pixel (c·W/cols, r·H/rows))."""

    t: float  # media time of the frame it was computed on
    mask: np.ndarray
    segment: int = 0  # the video segment of that frame (a seek or a loop starts a new one)
    view: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # the part of the frame the mask covers

    def road_beside(self, box: Box, width: int, height: int) -> float | None:
        """The share of road beside the box's feet: the more of two strips left and right of the box (from half a
        box width to 1.5 box widths out, just above its bottom edge), or None if neither strip is in the frame.

        Not under the box: YOLOPv2 leaves a hole in the road around a person (often wider than their box), and
        for someone close in front of us the bottom edge is our hood line, which is never road (review: people
        crossing a few metres ahead read as off the road). Someone standing in the road has road on at least one
        side; someone on a sidewalk has none within reach unless they stand right at the curb (then: on the road,
        the safe answer)."""
        rows, cols = self.mask.shape
        vx0, vy0, vx1, vy1 = self.view  # (a wide camera's lane model sees only the frame's center)
        ox, oy = vx0 * width, vy0 * height
        sx, sy = cols / ((vx1 - vx0) * width), rows / ((vy1 - vy0) * height)
        x1, y1, x2, y2 = box
        x1, x2, y1, y2 = x1 - ox, x2 - ox, y1 - oy, y2 - oy
        w, h = x2 - x1, y2 - y1
        r0 = max(0, math.floor((y2 - 0.12 * h) * sy))
        r1 = min(rows, math.ceil(y2 * sy))
        if r1 <= r0:
            return None
        best: float | None = None
        for lo, hi in ((x1 - 1.5 * w, x1 - 0.5 * w), (x2 + 0.5 * w, x2 + 1.5 * w)):  # (the hole is nearest)
            c0, c1 = max(0, math.floor(lo * sx)), min(cols, math.ceil(hi * sx))
            if c1 - c0 < 2:
                continue  # this side is out of the frame
            share = float(self.mask[r0:r1, c0:c1].mean())
            best = share if best is None else max(best, share)
        return best


@dataclass(frozen=True, slots=True)
class PersonView:
    """A pedestrian or cyclist as the path check saw it this frame."""

    track_id: int
    kind: str  # "pedestrian" | "cyclist" | "bicycle" (never seen with a rider: parked, not warned about)
    box: Box  # the box it was placed from (a cyclist's: person and bicycle together)
    x: float  # lateral position of its ground contact (m, + = right)
    z: float  # distance ahead (m)
    offset: float  # from our path's centreline (m, + = right)
    vx: float | None  # lateral speed (m/s, + = right; None without enough history)
    closing: float  # how fast we take it that we close in on it (m/s): sets how far the path reaches for it
    in_path: bool  # in the path zone this frame
    entering: bool  # outside it, but moving in within `predict_s`
    streak: int  # observations in a row in (or entering) the path
    alerted: bool
    ignored: str | None = None  # why it can't be warned about: "cut off" (side edge) | "parked bicycle" | "too tall"
    on_road: bool | None = None  # the ground at their feet is drivable (None: no fresh drivable-area mask)


@dataclass(frozen=True, slots=True)
class PedestrianEvent:
    """Someone in, or stepping into, our path while we move (spoken once per track)."""

    how: str  # "in_path" | "entering" | "crossing" (in front of us while we wait, not at a red light)
    t: float
    person: PersonView
    side: str  # "ahead" | "left" | "right" (for "entering": the side they come from)


@dataclass(frozen=True, slots=True)
class PedestrianSnapshot:
    """Read-only pedestrian state at one moment."""

    t: float
    zone: PathZone | None  # None before the first frame
    moving: bool  # we move (alerts may fire)
    still_reason: str | None  # why we count as not moving: "stopped" (ego motion) | "slow" (GPS)
    turning: bool
    people: tuple[PersonView, ...]
    aim_x: float = 0.0  # image column we're heading towards (the zone's centreline far away)
    horizon: float = 0.0  # horizon row used (px)
    focal_px: float = 0.0  # focal length used (px): ground (x, z) -> image (aim_x + f·x/z, horizon + f·h/z)


EMPTY = PedestrianSnapshot(0.0, None, False, "stopped", False, ())


@dataclass(frozen=True, slots=True)
class _Placed:
    """A box placed on the ground."""

    x: float  # lateral, from our heading (m, + = right)
    z: float  # distance ahead (m)
    clipped: bool  # cut off at a side edge of the frame
    tall: bool  # much taller than a person standing where it touches the ground


@dataclass(slots=True)
class _Rider:
    """Who rides a bicycle track."""

    person: int  # the rider's (person) track id
    frames: int = 0  # frames seen riding it
    last_t: float = 0.0  # last seen riding it


@dataclass(slots=True)
class _Person:
    """Per-track history and alert state."""

    kind: str
    samples: deque[tuple[float, float, float]] = field(default_factory=deque)  # (t, x, z)
    last_seen: float = 0.0
    streak: int = 0
    alerted: bool = False
    bike_frames: int = 0  # frames seen together with a bicycle (2 make it a cyclist for good)
    vx: float | None = None
    vz: float | None = None


def _overlap(a: Box, b: Box) -> float:
    """Area of a ∩ b over the area of a."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return 0.0 if w <= 0 or h <= 0 or area <= 0 else w * h / area


def _union(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _slope(ts: Sequence[float], vs: Sequence[float]) -> float:
    """Least-squares slope of vs over ts."""
    n = len(ts)
    mt, mv = sum(ts) / n, sum(vs) / n
    den = sum((t - mt) ** 2 for t in ts)
    return 0.0 if den <= 0 else sum((t - mt) * (v - mv) for t, v in zip(ts, vs, strict=True)) / den


class PedestrianTracker:
    """Per-frame path check for pedestrians and cyclists; `update` returns this frame's alert events."""

    def __init__(self, cfg: PedestriansConfig, hfov_deg: float, camera_height_m: float) -> None:
        self.cfg = cfg
        self.hfov_deg = hfov_deg
        self.camera_height_m = camera_height_m
        self.reset()

    def reset(self) -> None:
        """Forget everything (new video segment)."""
        self._people: dict[int, _Person] = {}
        self._rider_of: dict[int, _Rider] = {}  # bicycle track id -> who rides it
        self._yaw: deque[tuple[float, float]] = deque()  # (t, image shift this frame in °: + = we turn left)
        self._aim: deque[float] = deque(maxlen=self.cfg.aim_samples)  # per-frame focus-of-expansion columns
        self._aim_median: float | None = None  # cached median of _aim (None: recompute)
        self._last_t: float | None = None
        self._snapshot = EMPTY

    def snapshot(self) -> PedestrianSnapshot:
        """The latest state."""
        return self._snapshot

    # ------------------------------------------------------------------------------------ update

    def update(
        self,
        t: float,
        width: int,
        height: int,
        tracks: Sequence[Track],
        horizon: float,
        ego_stopped: bool,
        homography: np.ndarray | None = None,
        gps_speed: float | None = None,
        ego_confident: bool = True,
        drivable: DrivableArea | None = None,
        red_light: bool = False,
    ) -> list[PedestrianEvent]:
        """Fold one frame in. `horizon`: the horizon row (px); `gps_speed`: our GPS speed (m/s) with a fresh fix;
        `ego_confident`: ego motion could be measured (otherwise it holds its last state); `drivable`: YOLOPv2's
        drivable area of a recent frame (used if at most `road_max_age_s` old); `red_light`: our light reads red
        or yellow (people crossing in front of us while we wait there are normal: no `stopped_crossing` warning)."""
        cfg = self.cfg
        if self._last_t is not None and t < self._last_t - 1e-6:
            self.reset()  # time went backwards (a seek without a segment change): histories don't apply
        road = drivable if drivable is not None and 0.0 <= t - drivable.t <= cfg.road_max_age_s else None
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        cam = Camera.from_fov(width, self.hfov_deg, self.camera_height_m)

        if gps_speed is not None:
            speed = max(0.0, gps_speed)
            if ego_stopped and not ego_confident and speed >= cfg.gps_moving_mps:
                ego_stopped = False  # ego motion can't see (night, a blank road) and holds "stopped": GPS says moving
            still = "stopped" if ego_stopped else ("slow" if speed < cfg.min_speed_mps else None)
        else:
            speed = 0.0 if ego_stopped else cfg.default_speed_mps
            still = "stopped" if ego_stopped else None
        aim_x = self._update_aim(homography, cam, horizon, height, dt, ego_stopped, gps_speed)
        turn_rate = 0.0 if ego_stopped else self._turn_rate(t, homography, cam, aim_x, horizon)  # °/s, + = right
        turning = abs(turn_rate) >= cfg.turn_dps
        # The path's curvature: our turn rate over our speed, less what the reading can be off by (it bends gentle
        # curves partly, straight roads not at all), at most a tight turn.
        v_turn = max(speed, cfg.turn_speed_floor_mps)
        measured = math.radians(turn_rate) / v_turn
        curvature = math.copysign(
            min(max(0.0, abs(measured) - cfg.curvature_uncertainty), 1 / cfg.min_turn_radius_m), measured
        )
        # Waiting, but not at a red light (a crosswalk, a stop sign, a roundabout's entry): someone crossing right in
        # front of us is worth a word before we go, so the path reaches `stopped_zone_m` and alerts may fire.
        crossing_watch = cfg.stopped_crossing and still is not None and not red_light
        zone = PathZone(
            half_width=cfg.half_width_m,
            min_length=max(cfg.min_length_m, cfg.stopped_zone_m) if crossing_watch else cfg.min_length_m,
            headway_s=cfg.headway_s,
            max_length=min(cfg.max_length_m, cfg.turn_max_length_m) if turning else cfg.max_length_m,
            speed=speed,
            curvature=curvature,
            turn_rate=curvature * v_turn,
            uncertainty=cfg.curvature_uncertainty,
            min_half_width=cfg.min_half_width_m,
        )

        events: list[PedestrianEvent] = []
        views: list[PersonView] = []
        candidates = list(self._candidates(t, tracks))
        present = {track_id for track_id, _, _, _ in candidates}
        for track_id, kind, box, body_h in candidates:
            placed = self._place(box, body_h, cam, aim_x, horizon, width, height)
            if placed is None:
                continue
            share = None if road is None else road.road_beside(box, width, height)
            on_road = None if share is None else share >= cfg.road_min_share
            view = self._observe(
                t, track_id, kind, box, placed, zone, turning, still is None, present, events, on_road, crossing_watch
            )
            views.append(view)
        self._drop_stale(t)
        self._snapshot = PedestrianSnapshot(
            t, zone, still is None, still, turning, tuple(views), aim_x=aim_x, horizon=horizon, focal_px=cam.f
        )
        return events

    # ----------------------------------------------------------------------------------- pieces

    def _candidates(self, t: float, tracks: Sequence[Track]) -> Iterable[tuple[int, str, Box, float]]:
        """(track id, kind, box, body height in m) of every pedestrian and cyclist this frame."""
        cfg = self.cfg
        bikes = [tr for tr in tracks if tr.cls == "bicycle"]
        motos = [tr.box for tr in tracks if tr.cls == "motorcycle"]
        vehicles = [tr.box for tr in tracks if tr.cls in VEHICLE_CLASSES and tr.cls != "motorcycle"]
        ridden: set[int] = set()
        stood_in: set[int] = set()  # riders a bicycle stood in for this frame (once each)
        people = {tr.track_id for tr in tracks if tr.cls == "person"}
        for tr in tracks:
            if tr.cls != "person":
                continue
            if any(self._rides(tr.box, m) for m in motos):
                continue  # a motorcyclist: a vehicle in traffic
            if any(self._inside(tr.box, v) for v in vehicles):
                continue  # in a car or bus, or hidden behind one (no ground contact to place it by)
            bike = max(bikes, key=lambda b: _overlap(tr.box, b.box), default=None)
            person = self._people.get(tr.track_id)
            if bike is not None and self._rides(tr.box, bike.box):
                ridden.add(bike.track_id)
                rider = self._rider_of.get(bike.track_id)
                if rider is None or rider.person != tr.track_id:
                    rider = self._rider_of[bike.track_id] = _Rider(tr.track_id)
                rider.frames += 1
                rider.last_t = t
                if person is not None:
                    person.bike_frames += 1
                yield tr.track_id, CYCLIST, _union(tr.box, bike.box), cfg.cyclist_height_m
            elif person is not None and person.bike_frames >= cfg.rider_frames:
                yield tr.track_id, CYCLIST, tr.box, cfg.person_height_m  # the bicycle wasn't detected this frame
            else:
                yield tr.track_id, PEDESTRIAN, tr.box, cfg.person_height_m
        for bike in bikes:
            if bike.track_id in ridden:
                continue
            rider = self._rider_of.get(bike.track_id)
            if rider is not None and (rider.person in people or t - rider.last_t > cfg.rider_memory_s):
                del self._rider_of[bike.track_id]  # its rider is seen off it, or long gone: it is parked now
                rider = None
            if rider is not None and rider.frames >= cfg.rider_frames:
                if rider.person not in stood_in:
                    # Its rider wasn't detected this frame: the bicycle stands in for them (same track, one alert).
                    stood_in.add(rider.person)
                    yield rider.person, CYCLIST, bike.box, cfg.bicycle_height_m
            else:
                # No rider (or only someone passing it for a frame): most likely parked (at a rack, a pole, the curb).
                # Tracked, not warned about.
                yield bike.track_id, BICYCLE, bike.box, cfg.bicycle_height_m

    def _rides(self, person: Box, ride: Box) -> bool:
        """The person sits on the bicycle / motorcycle: boxes overlap, the person's centre is over it, their head
        is above it, and their feet are not below its wheels (someone walking in front of it, nearer to us, has
        their feet lower in the image)."""
        cx = (person[0] + person[2]) / 2
        ride_h = ride[3] - ride[1]
        return (
            _overlap(person, ride) >= self.cfg.rider_overlap
            and ride[0] <= cx <= ride[2]
            and ride[1] > person[1]
            and person[3] <= ride[3] + 0.1 * ride_h
        )

    def _inside(self, person: Box, vehicle: Box) -> bool:
        """The person box lies mostly inside the vehicle's and ends clearly above its bottom edge."""
        vh = vehicle[3] - vehicle[1]
        return _overlap(person, vehicle) >= self.cfg.inside_vehicle and person[3] < vehicle[3] - 0.1 * vh

    def _place(
        self, box: Box, body_h: float, cam: Camera, aim_x: float, horizon: float, width: int, height: int
    ) -> _Placed | None:
        """Where the box's ground contact is, or None if it can't be placed. x is measured from the direction
        we're heading (image column `aim_x`), not from the camera's axis."""
        cfg = self.cfg
        m = cfg.edge_margin_px
        x1, y1, x2, y2 = box
        left, right, top, bottom = x1 <= m, x2 >= width - m, y1 <= m, y2 >= height - m
        if ((left or right) and (top or bottom)) or (left and right):
            return None  # the windshield mount, a reflection, someone right beside the door
        h = y2 - y1
        if h < cfg.min_box_px:
            return None  # too far away to place
        z_ground = cam.f * cam.height_m / (y2 - horizon) if not bottom and y2 - horizon > 2.0 else None
        z_height = cam.f * body_h / h if not top and not bottom else None
        tall = False
        if z_ground is not None and z_height is not None:
            w = cfg.height_weight
            z = math.exp(w * math.log(z_height) + (1.0 - w) * math.log(z_ground))
            # Too tall even if the horizon were `tall_horizon_slack_deg` higher (which brings the ground point nearer).
            slack = cam.f * math.tan(math.radians(cfg.tall_horizon_slack_deg))
            z_near = cam.f * cam.height_m / (y2 - horizon + slack)
            tall = z_ground >= cfg.tall_min_z_m and z_near > cfg.max_tall_ratio * z_height
        elif z_ground is not None:
            z = z_ground
        elif z_height is not None:
            z = z_height
        elif bottom:
            z = cam.f * cam.height_m / max(height - horizon, 1.0)  # feet below the frame: at most this far
        else:
            return None
        x = ((x1 + x2) / 2 - aim_x) * z / cam.f  # (small-angle: the camera's yaw off our heading is a few degrees)
        return _Placed(x, z, left or right, tall)

    def _update_aim(
        self,
        homography: np.ndarray | None,
        cam: Camera,
        horizon: float,
        height: int,
        dt: float,
        ego_stopped: bool,
        gps_speed: float | None,
    ) -> float:
        """The image column we're heading towards: the median focus of expansion while driving (image centre
        until enough frames are in). Per frame: the ground's sideways flow on two rows below the horizon is
        expansion × (x − x_foe) plus our rotation (the same on both rows), so their difference is zero at x_foe
        whatever we turn."""
        cfg = self.cfg
        driving = not ego_stopped and (gps_speed is None or gps_speed >= cfg.aim_min_speed_mps)
        if homography is not None and driving and dt > 0 and height > horizon:
            y1, y2 = horizon + 0.35 * (height - horizon), horizon + 0.6 * (height - horizon)
            xs = np.linspace(0.25 * cam.cx, 1.75 * cam.cx, 9)
            pts = np.array([[*xs, *xs], [y1] * 9 + [y2] * 9, [1.0] * 18])
            q = homography @ pts
            if np.all(np.abs(q[2]) > 1e-9):
                flow = q[0] / q[2] - pts[0]
                d = flow[9:] - flow[:9]  # row 2 minus row 1: rotation cancels, expansion × (x − x_foe) is left
                xc = xs - xs.mean()
                da = float((xc * (d - d.mean())).sum() / (xc * xc).sum())  # expansion difference per frame
                foe = float(xs.mean() - d.mean() / da) if da > 0 else -1.0
                if da / dt >= cfg.aim_min_expansion and 0.0 <= foe <= 2 * cam.cx:
                    self._aim.append(foe)
                    self._aim_median = None
        if len(self._aim) < cfg.aim_min_samples:
            return cam.cx
        if self._aim_median is None:
            self._aim_median = float(np.median(self._aim))
        limit = cam.f * math.tan(math.radians(cfg.max_aim_deg))
        return min(max(self._aim_median, cam.cx - limit), cam.cx + limit)

    def _turn_rate(self, t: float, homography: np.ndarray | None, cam: Camera, aim_x: float, horizon: float) -> float:
        """Our turn rate (°/s, + = right) over `turn_window_s`: the sideways image shift of the point at the horizon
        straight ahead of us (at the focus of expansion forward motion moves nothing sideways)."""
        if homography is not None:
            p = homography @ np.array([aim_x, horizon, 1.0])
            if abs(p[2]) > 1e-9:
                self._yaw.append((t, math.degrees(math.atan2(p[0] / p[2] - aim_x, cam.f))))
        while self._yaw and t - self._yaw[0][0] > self.cfg.turn_window_s:
            self._yaw.popleft()
        if len(self._yaw) < 2:
            return 0.0
        span = self._yaw[-1][0] - self._yaw[0][0]
        # The scene shifts left when we turn right: the yaw change is minus the image shift.
        return 0.0 if span <= 0 else -sum(d for _, d in list(self._yaw)[1:]) / span

    def _observe(
        self,
        t: float,
        track_id: int,
        kind: str,
        box: Box,
        placed: _Placed,
        zone: PathZone,
        turning: bool,
        moving: bool,
        present: set[int],
        events: list[PedestrianEvent],
        on_road: bool | None = None,
        crossing_watch: bool = False,
    ) -> PersonView:
        cfg = self.cfg
        x, z = placed.x, placed.z
        person = self._people.get(track_id)
        if person is None:
            person = self._people[track_id] = _Person(kind)
            person.alerted = self._relinked(t, kind, x, z, zone, present)
        elif t - person.last_seen > cfg.max_gap_s:
            person.streak = 0  # unseen for a while: the confirmation starts over
        if person.bike_frames >= 2 and kind == PEDESTRIAN:
            kind = CYCLIST
        person.kind = kind
        if person.samples and self._jumped(person.samples[-1], t, x, z, zone.speed):
            person.samples.clear()  # the box jumped (an occlusion ended, a re-detection): no speed across it
        person.last_seen = t
        person.samples.append((t, x, z))
        while person.samples and t - person.samples[0][0] > cfg.fit_s:
            person.samples.popleft()
        person.vx, person.vz = self._velocity(person)

        # The path reaches 2 s (`headway_s`) at our speed; someone moving away with us (a cyclist riding ahead) is
        # closed in on more slowly. A measured approach never makes it longer (far away it is mostly noise), and
        # gets `closing_margin_mps` of benefit of the doubt; until it is measured (a new track) it counts as that
        # slow, so someone first seen riding ahead of us isn't warned about before we know they're not coming closer.
        approach = 0.0 if person.vz is None else max(0.0, -person.vz)
        closing = min(zone.speed, approach + cfg.closing_margin_mps)
        offset = zone.offset(x, z)
        # Off the road (a sidewalk) is not our path, except near the path's middle: the gate is for people at
        # the corridor's edges, and someone right ahead of us is warned about whatever the mask says.
        off_road = on_road is False and abs(zone.offset(x, z)) >= cfg.road_gate_min_offset_m
        in_path = zone.contains(x, z, closing) and not off_road
        entering = False
        vx = person.vx
        predict_z = max(cfg.predict_max_z_m, cfg.predict_max_s * zone.speed)
        predictable = not in_path and not turning and not placed.clipped and 0.0 < z <= predict_z
        if predictable and vx is not None and abs(vx) <= cfg.max_lateral_mps and z <= zone.length(closing):
            toward = -math.copysign(1.0, offset) * zone.lateral_speed(vx, z)  # speed towards the centreline
            # What static things seem to do sideways: unmodelled curves (z·ω), the distance drifting (|x|·rate).
            floor = max(cfg.min_lateral_mps, math.radians(cfg.yaw_noise_dps) * z, cfg.depth_drift * abs(x))
            excess = toward - floor  # how fast they surely move towards the path
            entering = excess > 0.0 and abs(offset) - zone.half_width_at(z) <= excess * cfg.predict_s
        # A box cut off at a side edge is beside us (its centre unknown); a bicycle without a rider is parked; a
        # "person" much taller than one standing where it touches the ground is a tree or pole, or someone whose
        # legs are hidden behind a car.
        ignored = None
        if placed.clipped:
            ignored = "cut off"
        elif kind == BICYCLE:
            ignored = "parked bicycle"
        elif placed.tall:
            ignored = "too tall"
        person.streak = person.streak + 1 if ignored is None and (in_path or entering) else 0

        fire = (moving or (crossing_watch and in_path)) and not person.alerted and person.streak >= cfg.confirm_frames
        person.alerted = person.alerted or fire
        view = PersonView(
            track_id,
            kind,
            box,
            x,
            z,
            offset,
            person.vx,
            closing,
            in_path,
            entering,
            person.streak,
            person.alerted,
            ignored,
            on_road,
        )
        if fire and not moving:  # (waiting, not at a red light: someone crossing in front of us)
            events.append(PedestrianEvent(CROSSING, t, view, "ahead"))
        elif fire:
            if in_path:
                side = "ahead" if abs(offset) < cfg.ahead_m else ("right" if offset > 0 else "left")
            else:
                side = "right" if offset > 0 else "left"
            events.append(PedestrianEvent(IN_PATH if in_path else ENTERING, t, view, side))
        return view

    def _jumped(self, last: tuple[float, float, float], t: float, x: float, z: float, speed: float) -> bool:
        """The position moved more since the last observation than our motion and a person's could explain."""
        cfg = self.cfg
        t0, x0, z0 = last
        dt = t - t0
        if dt <= 0 or z0 <= 0 or z <= 0:
            return False
        expected = -speed * dt / z0  # (log) distance change from our own approach
        return abs(math.log(z / z0) - expected) > cfg.jump_rel or abs(x - x0) > cfg.max_lateral_mps * dt + 0.2

    def _velocity(self, person: _Person) -> tuple[float | None, float | None]:
        """(dx/dt, dz/dt) over the last `fit_s`, or None without enough history."""
        samples = person.samples
        if len(samples) < self.cfg.fit_min_samples or samples[-1][0] - samples[0][0] < 0.5 * self.cfg.fit_s:
            return None, None
        ts = [s[0] for s in samples]
        return _slope(ts, [s[1] for s in samples]), _slope(ts, [s[2] for s in samples])

    def _relinked(self, t: float, kind: str, x: float, z: float, zone: PathZone, present: set[int]) -> bool:
        """A new track of the same kind where an alerted one was lost moments ago (ByteTrack gave the same person a
        new id). Tracks seen this frame are other people (someone walking next to a warned person is not them)."""
        cfg = self.cfg
        for track_id, old in self._people.items():
            dt = t - old.last_seen
            if track_id in present or not old.alerted or old.kind != kind or not 0.0 < dt <= cfg.relink_s:
                continue
            if not old.samples:
                continue
            _, ox, oz = old.samples[-1]
            closing = zone.speed if old.vz is None else -old.vz
            px, pz = ox + (old.vx or 0.0) * dt, oz - closing * dt
            if abs(x - px) <= cfg.relink_m and abs(z - pz) <= max(cfg.relink_m, 0.15 * z):
                return True
        return False

    def _drop_stale(self, t: float) -> None:
        keep = max(self.cfg.relink_s, self.cfg.fit_s)
        stale = [k for k, p in self._people.items() if t - p.last_seen > keep]
        for k in stale:
            del self._people[k]
        gone = [b for b, r in self._rider_of.items() if t - r.last_t > self.cfg.rider_memory_s]
        for b in gone:
            del self._rider_of[b]
