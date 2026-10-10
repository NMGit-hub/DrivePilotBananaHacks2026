"""Pedestrian and cyclist warnings (state/pedestrians.py) on simulated streets, and how they are spoken.

People are 3D boxes (1.7 m tall, 0.5 m wide; a cyclist is a rider box over a bicycle box) placed on a straight
road in world coordinates and projected through a pinhole dashcam (1280x720, 60° field of view, 1.3 m high,
horizon at row 360) while we drive along the road. Boxes jitter by ~1 px and detections can drop out; the tracker
gets nothing but the boxes (plus our GPS speed and stopped flag), like in the app.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest

from answers.alerts import PEDESTRIAN_TEXTS, AlertEngine
from answers.speech import Priority, Utterance
from core.config import AlertsConfig, PedestriansConfig, load_config
from core.types import Frame
from perception.detector import Track
from perception.ego_motion import NOT_MEASURED, EgoMotion
from perception.pipeline import PerceptionResult
from state.parking import Camera
from state.pedestrians import ENTERING, IN_PATH, PathZone, PedestrianEvent, PedestrianTracker
from state.world import WorldEvent, WorldState

W, H = 1280, 720
HFOV = 60.0
CAM_H = 1.3
CAM = Camera.from_fov(W, HFOV, CAM_H)
HORIZON = 360.0
FPS = 30.0
Box = tuple[float, float, float, float]


def to_camera(x: float, z: float, yaw_deg: float) -> tuple[float, float]:
    """A point in our heading's frame as the camera sees it when it is aimed `yaw_deg` right of our heading."""
    a = math.radians(yaw_deg)
    return x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)


def box_of(x: float, z: float, bottom: float, top: float, width: float, yaw_deg: float = 0.0) -> Box | None:
    """Image box of an upright object (bottom..top m above the ground) centred at (x, z) in our heading's frame."""
    pts = []
    for dx in (-width / 2, width / 2):
        for dz in (-0.2, 0.2):
            xc, zc = to_camera(x + dx, z + dz, yaw_deg)
            if zc <= 0.8:
                return None
            pts += [CAM.project(xc, zc, up, HORIZON) for up in (bottom, top)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, y1, x2, y2 = max(min(xs), 0.0), max(min(ys), 0.0), min(max(xs), W - 1.0), min(max(ys), H - 1.0)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2, y2


def turn_matrix(angle: float) -> np.ndarray:
    """Camera coordinates after we turn right by `angle` (rad): what was ahead is now to the left."""
    return np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])


def ground_homography(dz: float, yaw_deg: float = 0.0, radius: float | None = None, misread: float = 0.0) -> np.ndarray:
    """Frame-to-frame homography of the ground when we drive dz m along our heading (camera yawed `yaw_deg`), on a
    straight road or one curving right with `radius` (we turn by dz / radius); `misread` (rad) adds a turn that
    isn't there (a biased reading)."""
    a = math.radians(yaw_deg)
    k = np.array([[CAM.f, 0, CAM.cx], [0, CAM.f, HORIZON], [0, 0, 1.0]])
    n = np.array([0.0, 1.0, 0.0])  # the ground's normal (image rows grow downwards)
    bias = k @ turn_matrix(misread) @ np.linalg.inv(k)
    if radius is None:
        t = np.array([-dz * math.sin(a), 0.0, dz * math.cos(a)])  # our motion, in camera coordinates
        return bias @ k @ (np.eye(3) - np.outer(t, n) / CAM_H) @ np.linalg.inv(k)
    turn = dz / radius
    t = np.array([radius * (1 - math.cos(turn)), 0.0, radius * math.sin(turn)])
    return bias @ k @ turn_matrix(turn) @ (np.eye(3) - np.outer(t, n) / CAM_H) @ np.linalg.inv(k)


@dataclass
class Walker:
    """Someone on the street: world position (x across, z along the road) as a function of time."""

    track_id: int
    pos: Callable[[float], tuple[float, float]]
    kind: str = "pedestrian"  # "pedestrian" | "cyclist" | "bicycle" (no rider) | "motorcyclist" | "tree"
    visible: Callable[[float], bool] = lambda t: True
    hip_hidden_until: float = -1.0  # lower body hidden (behind a parked car) until this time...
    hidden_below_m: float = 0.9  # ...up to this height


class Street:
    """Drives along a straight road at a given speed; feeds the tracker frame by frame."""

    def __init__(
        self,
        walkers: list[Walker],
        cfg: PedestriansConfig | None = None,
        seed: int = 0,
        yaw_deg: float = 0.0,
        homographies: bool = False,
        jitter: float = 1.0,
        hood_row: float | None = None,
        radius: float | None = None,
        misread_dps: float = 0.0,
        horizon_error: float = 0.0,
    ) -> None:
        self.horizon_error = horizon_error  # the tracker is told the horizon is this many px lower than it is
        self.confident = True  # ego motion could be measured
        self.radius = radius  # the road curves right with this radius (None: straight)
        self.misread_dps = misread_dps  # the homography reads a turn (°/s, + = right) that isn't there
        self.walkers = walkers
        self.tracker = PedestrianTracker(cfg or PedestriansConfig(), HFOV, CAM_H)
        self.rng = np.random.default_rng(seed)
        self.yaw = yaw_deg
        self.homographies = homographies
        self.jitter = jitter
        self.hood_row = hood_row  # our hood hides everything below this image row
        self.t = 0.0
        self.s = 0.0  # how far we have driven (m)
        self.events: list[PedestrianEvent] = []
        self.extra: list[Track] = []  # boxes added as they are (artifacts)

    def ahead(self, across: float, along: float) -> tuple[float, float]:
        """A road point (`across` m right of the road's centre line, `along` m along it) relative to us: (x, z)."""
        if self.radius is None:
            return across, along - self.s
        r = self.radius

        def point(a: float, d: float) -> tuple[float, float]:  # world (east, north); the arc's centre is (r, 0)
            return r - (r - d) * math.cos(a / r), (r - d) * math.sin(a / r)

        (ox, oy), (px, py), phi = point(self.s, 0.0), point(along, across), self.s / r
        rx, ry = px - ox, py - oy
        return rx * math.cos(phi) - ry * math.sin(phi), rx * math.sin(phi) + ry * math.cos(phi)

    def tracks(self) -> list[Track]:
        out: list[Track] = []
        for w in self.walkers:
            if not w.visible(self.t):
                continue
            wx, z = self.ahead(*w.pos(self.t))
            boxes: list[tuple[int, str, Box | None]] = []
            low = w.hidden_below_m if self.t < w.hip_hidden_until else 0.0
            if w.kind in ("pedestrian", "tree"):
                top = 3.0 if w.kind == "tree" else 1.7
                boxes.append((w.track_id, "person", box_of(wx, z, low, top, 0.5, self.yaw)))
            elif w.kind in ("cyclist", "motorcyclist"):
                ride = "bicycle" if w.kind == "cyclist" else "motorcycle"
                boxes.append((w.track_id, "person", box_of(wx, z, 0.35, 1.75, 0.5, self.yaw)))
                boxes.append((w.track_id + 5000, ride, box_of(wx, z, 0.0, 1.05, 0.6, self.yaw)))
            elif w.kind == "bicycle":
                boxes.append((w.track_id, "bicycle", box_of(wx, z, 0.0, 1.05, 0.6, self.yaw)))
            for track_id, cls, box in boxes:
                if box is None:
                    continue
                if self.hood_row is not None:
                    box = (box[0], box[1], box[2], min(box[3], self.hood_row))
                j = self.rng.normal(0.0, self.jitter, 4)
                jittered = tuple(float(v) for v in np.add(box, j))
                out.append(Track(track_id, cls, 0.8, jittered))  # type: ignore[arg-type]
        return out + self.extra

    def frame(self, speed: float, stopped: bool = False, gps: float | str | None = "speed") -> list[PedestrianEvent]:
        """One frame; then we move on by speed / FPS. `gps`: "speed" = our speed, None = no fix, or a value."""
        dz = speed / FPS
        misread = math.radians(self.misread_dps) / FPS
        hom = ground_homography(dz, self.yaw, self.radius, misread) if self.homographies and dz > 0 else None
        gps_speed = speed if gps == "speed" else gps
        assert gps_speed is None or isinstance(gps_speed, float | int)
        horizon = HORIZON + self.horizon_error
        events = self.tracker.update(self.t, W, H, self.tracks(), horizon, stopped, hom, gps_speed, self.confident)
        self.events += events
        self.t = round(self.t + 1 / FPS, 6)
        self.s += dz
        return events

    def drive(self, seconds: float, speed: float, stopped: bool = False) -> None:
        for _ in range(round(seconds * FPS)):
            self.frame(speed, stopped)

    def kinds(self) -> list[tuple[int, str, str]]:
        return [(e.person.track_id, e.how, e.side) for e in self.events]


def standing(x: float, z: float) -> Callable[[float], tuple[float, float]]:
    return lambda t: (x, z)


def walking(
    x0: float, z0: float, vx: float, vz: float = 0.0, start: float = 0.0
) -> Callable[[float], tuple[float, float]]:
    """Standing at (x0, z0) until `start`, then walking at (vx, vz) m/s."""
    return lambda t: (x0 + vx * max(0.0, t - start), z0 + vz * max(0.0, t - start))


# ---------------------------------------------------------------------------------------------- the zone


def test_the_zone_reaches_further_the_faster_we_go_and_bends_only_with_curvature() -> None:
    zone = PathZone(half_width=1.5, min_length=6.0, headway_s=2.0, max_length=35.0, speed=10.0)
    assert zone.length() == pytest.approx(26.0)
    assert zone.length(closing=0.0) == pytest.approx(6.0)  # someone moving away with us at our speed
    assert PathZone(1.5, 6.0, 2.0, 35.0, speed=30.0).length() == pytest.approx(35.0)  # capped
    assert zone.contains(1.4, 20.0) and not zone.contains(1.6, 20.0) and not zone.contains(0.0, 27.0)
    bent = replace(zone, curvature=1 / 20)  # a right turn of 20 m radius
    assert bent.center(10.0) == pytest.approx(2.5)
    assert bent.contains(3.0, 10.0) and not bent.contains(-1.0, 10.0)


# ------------------------------------------------------------------------------------- crossing in front


@pytest.mark.parametrize("seed", range(4))
def test_someone_crossing_in_front_while_we_drive_is_warned_about_once_before_they_reach_our_path(seed: int) -> None:
    # 40 m ahead, from the right sidewalk (4.5 m out), walking left at 1.4 m/s from t=1 s; we drive at 8 m/s.
    walker = Walker(7, walking(4.5, 40.0, -1.4, start=1.0))
    street = Street([walker], seed=seed)
    street.drive(4.4, 8.0)
    assert len(street.events) == 1
    event = street.events[0]
    assert event.person.track_id == 7 and event.person.kind == "pedestrian"
    assert event.side == "right"
    assert 4.5 - 1.4 * (event.t - 1.0) > 1.0  # warned while they are still beside our car's path, not in front
    if event.how == ENTERING:
        assert event.person.offset > 1.0  # predicted: not in the path yet
    utterance = AlertEngine(AlertsConfig(), lambda u: None).on_world_event(
        WorldEvent("pedestrian_in_path", event.t, {"track": 7, "cls": "pedestrian", "how": event.how, "side": "right"})
    )
    assert utterance is not None and utterance.text in (
        "Pedestrian stepping into the road on your right.",
        "Pedestrian ahead on your right.",
    )


def test_someone_stepping_out_close_ahead_is_predicted_before_they_are_in_the_path() -> None:
    # Standing at the curb 2.6 m out, 30 m ahead; steps out at 1.5 m/s when we are ~12 m away (we drive 6 m/s).
    street = Street([Walker(3, walking(2.6, 30.0, -1.5, start=3.0))])
    street.drive(3.8, 6.0)
    assert street.kinds() == [(3, ENTERING, "right")]
    assert street.events[0].t < 3.0 + (2.6 - 1.5) / 1.5  # before reaching our path


# -------------------------------------------------------------------------------- things that must not alert


@pytest.mark.parametrize("seed", range(4))
def test_people_walking_along_the_sidewalks_are_not_warned_about(seed: int) -> None:
    walkers = [
        Walker(1, walking(3.2, 20.0, 0.0, -1.4)),  # right sidewalk, towards us
        Walker(2, walking(3.5, 15.0, 0.0, 1.4)),  # right sidewalk, away from us
        Walker(3, walking(-3.6, 25.0, 0.0, -1.2)),  # left sidewalk
        Walker(4, walking(2.8, 45.0, -0.1, -1.0)),  # drifting slowly towards the curb, but not into the road
    ]
    street = Street(walkers, seed=seed)
    street.drive(6.0, 10.0)
    assert street.events == []


@pytest.mark.parametrize("speed", [4.0, 10.0, 15.0])
def test_people_standing_at_the_curb_are_not_warned_about(speed: float) -> None:
    walkers = [Walker(1, standing(2.4, 30.0)), Walker(2, standing(-2.5, 22.0)), Walker(3, standing(2.7, 50.0))]
    for seed in range(3):
        street = Street(walkers, seed=seed, jitter=1.5)
        street.drive(4.0, speed)
        assert street.events == [], (speed, seed, street.kinds())


def test_people_crossing_while_we_wait_at_a_red_light_are_not_warned_about() -> None:
    # Stopped for 12 s; three people cross 5-9 m in front of us, both ways; then the light turns green after
    # they have crossed and we drive off.
    walkers = [
        Walker(1, walking(5.0, 7.0, -1.3, start=1.0)),
        Walker(2, walking(-5.0, 8.0, 1.2, start=2.0)),
        Walker(3, walking(4.5, 6.0, -1.5, start=4.0)),
    ]
    street = Street(walkers)
    street.drive(14.0, 0.0, stopped=True)
    assert street.events == []
    for k in range(round(3.0 * FPS)):  # pulling away (they're on the far sidewalks by now)
        street.frame(min(8.0, 2.0 * k / FPS))
    assert street.events == []


def test_someone_still_in_our_path_when_we_start_moving_is_warned_about_at_once() -> None:
    # A slow walker is crossing 5 m in front of us when we start (GPS speed ramps up at 2 m/s²).
    street = Street([Walker(5, walking(3.0, 5.0, -0.5, start=0.0))])
    street.drive(5.0, 0.0, stopped=True)  # they are in front of us now (x = 0.5)
    assert street.events == []
    start = street.t
    for k in range(round(2.0 * FPS)):
        street.frame(2.0 * k / FPS)
    assert street.kinds() == [(5, IN_PATH, "ahead")]
    # Warned as soon as the GPS speed reaches `min_speed_mps` (0.5 m/s after 0.25 s), within a frame of it.
    assert street.events[0].t - start <= 0.25 + 2 / FPS


def test_a_one_frame_false_detection_in_our_path_says_nothing() -> None:
    street = Street([Walker(9, standing(0.0, 15.0), visible=lambda t: 1.0 <= t < 1.0 + 2 / FPS)])
    street.drive(3.0, 8.0)
    assert street.events == []
    # Nor one that flickers up for a frame every half second (the streak restarts after each gap).
    flicker = Street([Walker(9, standing(0.0, 12.0), visible=lambda t: round(t * FPS) % 15 == 0)])
    flicker.drive(5.0, 2.0)
    assert flicker.events == []


@pytest.mark.parametrize("gap_s", [0.2, 0.6])
def test_a_detection_dropout_does_not_warn_twice(gap_s: float) -> None:
    def gone(t: float) -> bool:
        return not 2.0 <= t < 2.0 + gap_s

    street = Street([Walker(4, standing(0.3, 30.0), visible=gone)])
    street.drive(3.0, 8.0)
    assert street.kinds() == [(4, IN_PATH, "ahead")]


def test_an_id_switch_after_a_dropout_does_not_warn_twice() -> None:
    # The same person is re-detected under a new track id after 0.5 s unseen.
    before = Walker(4, standing(0.3, 30.0), visible=lambda t: t < 2.0)
    after = Walker(41, standing(0.3, 30.0), visible=lambda t: t >= 2.5)
    street = Street([before, after])
    street.drive(3.2, 8.0)
    assert street.kinds() == [(4, IN_PATH, "ahead")]


def test_two_people_side_by_side_are_two_tracks_but_the_second_is_not_an_id_switch() -> None:
    street = Street([Walker(1, standing(0.2, 25.0)), Walker(2, standing(0.9, 25.5))])
    street.drive(2.5, 8.0)
    assert sorted(track for track, _, _ in street.kinds()) == [1, 2]  # (the alert engine makes them one warning)


# -------------------------------------------------------------------------------------------- cyclists


def test_a_cyclist_riding_ahead_of_us_at_our_speed_is_not_warned_about() -> None:
    # Decision: someone we aren't closing in on (riding ahead at our speed, 15 m out) is not a surprise to warn of.
    street = Street([Walker(6, lambda t: (0.3, 15.0 + 7.0 * t), kind="cyclist")])
    street.drive(8.0, 7.0)
    assert street.events == []


def test_a_cyclist_we_catch_up_with_is_warned_about_once() -> None:
    street = Street([Walker(6, lambda t: (0.6, 45.0 + 3.0 * t), kind="cyclist")])
    street.drive(6.0, 10.0)
    assert street.kinds() == [(6, IN_PATH, "right")]
    assert street.events[0].person.kind == "cyclist"
    assert street.events[0].person.z <= 6.0 + 2.0 * 10.0 + 0.5  # within the 2 s path at our speed


def test_a_cyclist_cutting_in_from_the_bike_lane() -> None:
    street = Street([Walker(8, lambda t: (2.4 - 1.2 * max(0.0, t - 2.0), 20.0 + 4.0 * t), kind="cyclist")])
    street.drive(3.5, 8.0)
    assert [(track, side) for track, _, side in street.kinds()] == [(8, "right")]
    assert street.events[0].person.kind == "cyclist"


def test_parked_bicycles_and_motorcyclists_are_not_warned_about() -> None:
    walkers = [
        Walker(1, standing(0.8, 20.0), kind="bicycle"),  # left in the road, no rider: parked
        Walker(2, lambda t: (0.2, 12.0 + 8.0 * t), kind="motorcyclist"),  # a motorcycle ahead in traffic
    ]
    street = Street(walkers)
    street.drive(4.0, 8.0)
    assert street.events == []


def test_someone_in_front_of_a_motorcycle_is_not_taken_for_its_rider() -> None:
    # Waiting at a light behind a motorcyclist (11 m); someone stands between us (9 m), their box over the
    # motorcycle's in the image. When we move off, they are warned about; the rider is not.
    walkers = [Walker(1, standing(0.0, 11.0), kind="motorcyclist"), Walker(2, standing(0.1, 9.0))]
    street = Street(walkers)
    street.drive(1.0, 0.0, stopped=True)
    for k in range(round(1.0 * FPS)):
        street.frame(1.5 * k / FPS)
    assert street.kinds() == [(2, IN_PATH, "ahead")]


# --------------------------------------------------------------------------------- detector look-alikes


def test_a_tree_taken_for_a_person_is_not_warned_about() -> None:
    # A 3 m tall "person" (a trunk) in our path, 25 m out down to 9 m: not someone standing in the road.
    tree = Street([Walker(1, standing(0.0, 25.0), kind="tree")])
    tree.drive(2.0, 8.0)
    assert tree.events == []
    assert any(p.ignored == "too tall" for p in tree.tracker.snapshot().people)


def test_someone_right_in_front_of_our_hood_is_warned_about_when_we_start_moving() -> None:
    # Feet hidden by the hood (below row 690): the ground-contact distance is too far, the box too short. Up close
    # the "too tall" check doesn't apply (here made strict enough that it would otherwise reject them).
    cfg = PedestriansConfig(max_tall_ratio=1.05)
    near = Street([Walker(2, standing(0.3, 3.0))], cfg=cfg, hood_row=690.0)
    near.drive(1.0, 0.0, stopped=True)
    assert near.events == []
    for k in range(round(1.0 * FPS)):
        near.frame(2.0 * k / FPS)
    assert near.kinds() == [(2, IN_PATH, "ahead")]


@pytest.mark.parametrize("hidden_below", [0.4, 0.9])
def test_someone_whose_legs_come_into_view_is_not_stepping_into_the_road(hidden_below: float) -> None:
    # At the curb 2.5 m out, legs hidden behind a parked car until t=1.5: the box suddenly grows downwards and the
    # person seems to jump towards the road. That jump is not someone walking into it.
    street = Street([Walker(1, standing(2.5, 25.0), hip_hidden_until=1.5, hidden_below_m=hidden_below)])
    street.drive(3.0, 6.0)
    assert street.events == []


@pytest.mark.parametrize("bottom", [713.0, 719.0])
def test_the_windshield_mount_at_the_frame_edge_is_ignored_even_in_a_turn(bottom: float) -> None:
    # Glued to the right edge (713: as on the rain dev clip, ignored as cut off at a side; 719: in the bottom
    # corner too, never placed).
    street = Street([])
    street.extra = [Track(77, "person", 0.7, (1023.0, 276.0, 1279.0, bottom))]
    a = math.radians(-20.0 / FPS)  # turning right at 20°/s (the scene turns left)
    k = np.array([[CAM.f, 0, CAM.cx], [0, CAM.f, HORIZON], [0, 0, 1.0]])
    r = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    turn = k @ r @ np.linalg.inv(k)
    for _ in range(round(3.0 * FPS)):
        street.events += street.tracker.update(street.t, W, H, street.tracks(), HORIZON, False, turn, 5.0)
        street.t += 1 / FPS
    snap = street.tracker.snapshot()
    assert snap.turning
    assert street.events == []
    if bottom < 716.0:
        assert [p.ignored for p in snap.people] == ["cut off"]
    else:
        assert snap.people == ()


# ------------------------------------------------------------------------------------ where we're heading


def test_a_camera_aimed_off_our_heading_is_corrected_for() -> None:
    # The dashcam looks 6° right of where the car goes: "straight ahead in the image" is the right sidewalk 25 m
    # out. Someone standing there (3 m right of our path) must not be in our path; someone in it must be.
    expected_aim = CAM.cx - CAM.f * math.tan(math.radians(6.0))
    walkers = [Walker(1, standing(3.0, 120.0)), Walker(2, standing(0.0, 125.0), visible=lambda t: t > 4.0)]
    street = Street(walkers, yaw_deg=6.0, homographies=True, jitter=0.5)
    street.drive(15.0, 8.0)
    assert street.tracker.snapshot().aim_x == pytest.approx(expected_aim, abs=10.0)
    assert street.kinds() == [(2, IN_PATH, "ahead")]
    # Without the correction (never enough frames to estimate it) the sidewalk person would be "in our path".
    blind = Street(walkers, cfg=PedestriansConfig(aim_min_samples=100_000), yaw_deg=6.0, homographies=True)
    blind.drive(15.0, 8.0)
    assert (1, IN_PATH) in [(track, how) for track, how, _ in blind.kinds()]


def test_on_a_gentle_curve_the_path_bends_and_the_outside_curb_stays_out_of_it() -> None:
    # The road curves right (radius 150 m) at 10 m/s, 3.8°/s. Looking straight ahead, the far (left) curb crosses
    # our view 20-30 m out; people standing there are not in our path. Someone standing in our lane is.
    walkers = [
        *(Walker(i, standing(-2.4, a)) for i, a in ((1, 60.0), (2, 100.0), (3, 140.0))),  # outside curb
        *(Walker(i, standing(2.4, a)) for i, a in ((4, 80.0), (5, 120.0))),  # inside curb
        Walker(6, standing(0.0, 160.0)),  # in our lane
    ]
    street = Street(walkers, homographies=True, radius=150.0)
    street.drive(15.0, 10.0)
    assert [track for track, _, _ in street.kinds()] == [6]
    assert street.events[0].person.z > 15.0  # still well ahead
    # A path that doesn't bend (no homography: no turn rate) would take the outside curb for our lane.
    straight = Street(walkers, cfg=PedestriansConfig(curvature_uncertainty=0.0), radius=150.0)
    straight.drive(15.0, 10.0)
    assert {1, 2, 3} & {track for track, _, _ in straight.kinds()}


def test_a_turn_rate_misread_on_a_straight_road_does_not_bend_the_path_into_the_curb() -> None:
    # The homography reads 2.5°/s to the right on a straight road (the dashcam bias seen on the dev clips). Someone
    # standing at the right curb 2.2 m out, ahead at 20-30 m, must not end up in a path bent towards them.
    walkers = [Walker(1, standing(2.2, 40.0)), Walker(2, standing(2.2, 60.0))]
    street = Street(walkers, homographies=True, misread_dps=2.5)
    street.drive(4.0, 10.0)
    assert street.events == []
    trusting = Street(walkers, cfg=PedestriansConfig(curvature_uncertainty=0.0), homographies=True, misread_dps=2.5)
    trusting.drive(4.0, 10.0)
    assert trusting.events != []  # (taking the reading at face value would)


@pytest.mark.parametrize("seed", range(3))
def test_a_slow_sideways_shuffle_at_the_curb_is_not_a_step_into_the_road(seed: int) -> None:
    # 3 m out, 25 m ahead (we drive 6 m/s): shuffling towards the road at 0.6 m/s is within what distance drift makes
    # static people seem to do that far out (|x| × 0.25/s); stepping out at 1.6 m/s is not.
    shuffle = Street([Walker(1, walking(3.0, 25.0, -0.6, start=1.0), visible=lambda t: t < 3.0)], seed=seed)
    shuffle.drive(3.0, 6.0)
    assert shuffle.events == []
    step = Street([Walker(2, walking(3.0, 25.0, -1.6, start=1.0))], seed=seed)
    step.drive(2.6, 6.0)
    assert [(track, how) for track, how, _ in step.kinds()] == [(2, ENTERING)]


# ------------------------------------------------------------------------------------------- robustness


def test_no_gps_falls_back_to_ego_motion_and_a_city_speed() -> None:
    street = Street([Walker(3, standing(0.2, 40.0))])
    for _ in range(round(3.0 * FPS)):
        street.frame(8.0, gps=None)
    assert street.kinds() == [(3, IN_PATH, "ahead")]
    # Stopped by ego motion (no GPS): nothing, even right in front.
    stopped = Street([Walker(3, standing(0.2, 5.0))])
    for _ in range(round(3.0 * FPS)):
        stopped.frame(0.0, stopped=True, gps=None)
    assert stopped.events == []


def test_gps_says_we_are_barely_moving_while_ego_motion_flickers() -> None:
    # Waiting at a light, ego motion flickers to "moving" (a bus crossing) while GPS says 0.1 m/s.
    street = Street([Walker(1, walking(3.0, 6.0, -1.0, start=0.5))])
    for k in range(round(6.0 * FPS)):
        street.frame(0.0, stopped=(k // 15) % 2 == 0, gps=0.1)
    assert street.events == []


def test_time_going_backwards_starts_over() -> None:
    street = Street([Walker(4, standing(0.3, 30.0))])
    street.drive(3.0, 8.0)
    assert len(street.events) == 1
    street.t, street.s = 0.0, 0.0  # replay seek back without a segment change
    street.drive(3.0, 8.0)
    # The tracker starts over and raises the event again. (In the app every seek or loop also starts a new segment,
    # which resets the alert engine's memory of who was warned, so it is spoken again.)
    assert len(street.events) == 2


def test_the_update_is_cheap() -> None:
    walkers = [Walker(i, walking(-6.0 + 1.3 * i, 15.0 + 3.0 * i, 0.3, -1.0)) for i in range(10)]
    street = Street(walkers, homographies=True)
    started = time.perf_counter()
    street.drive(10.0, 8.0)
    per_frame_ms = (time.perf_counter() - started) * 1000.0 / (10.0 * FPS)
    assert per_frame_ms < 5.0  # includes building the boxes; the update itself measures ~0.1 ms on the dev clips


# ----------------------------------------------------------------------- independent review regressions


def spoken(street: Street) -> list[str]:
    """What the alert engine says for the street's events."""
    alerts, said = engine()
    for e in street.events:
        detail = {"track": e.person.track_id, "cls": e.person.kind, "how": e.how, "side": e.side}
        alerts.on_world_event(WorldEvent("pedestrian_in_path", e.t, detail))
    return [u.text for u in said]


def test_a_second_hazard_moments_after_a_warning_is_still_spoken() -> None:
    # Someone standing in our lane 30 m ahead is warned about; 1.5 s later a cyclist cuts in from the right: a new
    # hazard, not the same group, so it is said too.
    a = Walker(1, standing(-0.6, 30.0))
    b = Walker(2, lambda t: (2.6 - 1.5 * max(0.0, t - 1.6), 28.0 + 3.0 * t), kind="cyclist")
    street = Street([a, b])
    street.drive(3.5, 8.0)
    texts = spoken(street)
    assert len(texts) == 2 and texts[0].startswith("Pedestrian") and texts[1].startswith("Cyclist")


def test_a_parked_bicycle_someone_stood_behind_is_not_a_cyclist() -> None:
    # A bicycle left 0.9 m right of our path, 30 m ahead; someone stands just behind it (overlapping it in the image)
    # for 0.3 s, then their track is lost. Nobody rides it: no warning when we reach it.
    bike = Walker(1, standing(0.9, 30.0), kind="bicycle")
    walker = Walker(2, standing(1.2, 30.3), visible=lambda t: t < 0.3)
    street = Street([bike, walker])
    street.drive(3.0, 8.0)
    assert street.events == []


@pytest.mark.parametrize(("error_px", "warned_by_m"), [(25.0, 18.0), (35.0, 14.0)])
def test_a_horizon_estimated_too_low_does_not_hide_someone_in_our_path(error_px: float, warned_by_m: float) -> None:
    # Braking pitch or a road rising ahead: the horizon is 1.3-1.8° lower than estimated. The ground-contact
    # distance comes out too far and the person seems too tall for it, but they are still warned about in time.
    street = Street([Walker(1, standing(0.0, 40.0))], horizon_error=error_px)
    street.drive(3.5, 10.0)
    assert len(street.events) == 1
    assert 40.0 - 10.0 * street.events[0].t >= warned_by_m  # true distance when warned


def test_a_cyclist_appearing_where_a_warned_pedestrian_was_lost_is_not_taken_for_them() -> None:
    ped = Walker(1, standing(0.2, 30.0), visible=lambda t: t < 2.0)
    cyclist = Walker(2, lambda t: (-0.6, 30.0 + 2.0 * (t - 2.6)), kind="cyclist", visible=lambda t: t >= 2.6)
    street = Street([ped, cyclist])
    street.drive(3.4, 8.0)
    assert [track for track, _, _ in street.kinds()] == [1, 2]


def test_ego_motion_that_cannot_see_does_not_mute_warnings_while_gps_says_we_drive() -> None:
    # At night ego motion can't be measured and holds "stopped" from the last light; GPS says 9 m/s.
    street = Street([Walker(1, standing(0.3, 30.0))])
    street.confident = False
    street.drive(3.0, 9.0, stopped=True)
    assert street.kinds() == [(1, IN_PATH, "ahead")]
    # (with a confident "stopped" nothing is said, whatever GPS says: GPS lags when we stop)
    waiting = Street([Walker(1, standing(0.3, 5.0))])
    for _ in range(round(2.0 * FPS)):
        waiting.frame(0.0, stopped=True, gps=2.5)
    assert waiting.events == []


# --------------------------------------------------------------------------------------- the alert engine


def engine(**cfg: float) -> tuple[AlertEngine, list[Utterance]]:
    said: list[Utterance] = []
    return AlertEngine(AlertsConfig(**cfg), said.append), said


def ped(t: float, track: int, cls: str = "pedestrian", how: str = "in_path", side: str = "ahead") -> WorldEvent:
    return WorldEvent("pedestrian_in_path", t, {"track": track, "cls": cls, "how": how, "side": side})


def test_every_warning_is_a_fixed_pre_renderable_sentence() -> None:
    texts = set(PEDESTRIAN_TEXTS.values())
    assert len(texts) == len(PEDESTRIAN_TEXTS) == 10
    for (cls, _how, side), text in PEDESTRIAN_TEXTS.items():
        assert cls.capitalize() in text and (side == "ahead" or side in text)
    from app import FIXED_ALERT_TEXTS

    assert texts <= set(FIXED_ALERT_TEXTS)  # pre-rendered at startup: they play at once


def test_a_warning_is_safety_priority_and_kept_in_quiet_mode() -> None:
    alerts, said = engine()
    alerts.set_quiet(True)
    utterance = alerts.on_world_event(ped(10.0, 3, side="right", how="entering"))
    assert utterance is not None and said == [utterance]
    assert utterance.priority is Priority.SAFETY and utterance.kind == "alert"
    assert utterance.text == "Pedestrian stepping into the road on your right."
    assert not utterance.drops_answers  # an interrupted answer resumes afterwards


def test_once_per_track_and_one_warning_for_a_group() -> None:
    alerts, said = engine(safety_gap_s=3.0)
    alerts.on_world_event(ped(10.0, 3))
    alerts.on_world_event(ped(30.0, 3))  # the same person again much later (a re-detection): not again
    alerts.on_world_event(ped(10.5, 4, side="left"))  # someone next to them: one warning for the group
    alerts.on_world_event(ped(11.0, 5, how="entering", side="right"))  # ("ahead" covers either side)
    alerts.on_world_event(ped(14.0, 6, cls="cyclist", side="left"))  # a new hazard after the gap
    assert [u.text for u in said] == ["Pedestrian ahead.", "Cyclist ahead on your left."]


def test_within_the_gap_another_kind_or_the_other_side_is_a_new_warning() -> None:
    alerts, said = engine(safety_gap_s=3.0)
    alerts.on_world_event(ped(10.0, 1, how="entering", side="left"))
    alerts.on_world_event(ped(10.4, 2, how="entering", side="left"))  # the same group: once
    alerts.on_world_event(ped(10.8, 3, cls="cyclist", side="left"))  # a cyclist: said
    alerts.on_world_event(ped(11.2, 4, how="entering", side="right"))  # the other side: said
    assert [u.text for u in said] == [
        "Pedestrian stepping into the road on your left.",
        "Cyclist ahead on your left.",
        "Pedestrian stepping into the road on your right.",
    ]


def test_a_new_segment_forgets_who_was_warned_about() -> None:
    alerts, said = engine()
    alerts.on_world_event(ped(10.0, 3))
    alerts.reset()  # seek: track ids start over
    alerts.on_world_event(ped(5.0, 3))
    assert len(said) == 2


def test_an_unknown_phrase_says_nothing() -> None:
    alerts, said = engine()
    assert alerts.on_world_event(ped(1.0, 3, cls="horse")) is None and said == []


# ------------------------------------------------------------------------------------------ world state


def test_world_state_emits_the_warning_once_and_snapshots_the_people() -> None:
    events: list[WorldEvent] = []
    world = WorldState(load_config(overrides=["video.file.hfov_deg=60.0"]), on_event=events.append)
    street = Street([Walker(7, standing(0.4, 30.0))])
    image = np.full((H, W, 3), 100, np.uint8)
    for k in range(round(3.0 * FPS)):
        tracks = street.tracks()
        ego = EgoMotion(
            t=street.t,
            short=NOT_MEASURED,
            long=NOT_MEASURED,
            value=5.0,
            stopped=False,
            stopped_since=None,
            confident=True,
            homography=None,
            elapsed_ms=0.0,
        )
        frame = Frame(image=image, t=street.t, index=k, captured_at=0.0)
        world.on_perception(PerceptionResult(frame, tracks, ego, {}, new_segment=k == 0))
        street.t = round(street.t + 1 / FPS, 6)
        street.s += 8.0 / FPS
    warnings = [e for e in events if e.kind == "pedestrian_in_path"]
    assert len(warnings) == 1
    detail = warnings[0].detail
    assert detail["track"] == 7 and detail["cls"] == "pedestrian" and detail["how"] == "in_path"
    assert isinstance(warnings[0].payload, PedestrianEvent)
    snap = world.snapshot()
    assert snap is not None and [p.track_id for p in snap.pedestrians.people] == [7]
    assert snap.pedestrians.people[0].alerted
