"""Parking spots (state/parking.py) in simulated parking lots.

Cars are 3D boxes parked side by side along an aisle (both sides), projected through a pinhole dashcam (1280x720,
60° field of view, 1.3 m high) while we drive down the aisle. Heavily hidden cars are not "detected" (as YOLO
misses them), boxes jitter by ~1 px, and individual detections can drop out. The tracker gets nothing but those
boxes, like in the app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pytest

from core.config import ParkingConfig
from perception.detector import Track
from state.parking import Camera, ParkingEvent, ParkingTracker, coverage

W, H = 1280, 720
HFOV = 60.0
CAM = Camera.from_fov(W, HFOV, 1.3)
HORIZON = 360.0
FPS = 15.0
PITCH = 2.7
EDGE = 3.5  # aisle-side edge of each row (m from our path)


@dataclass(frozen=True)
class Car3D:
    """A world-space box: x (m, + right), z (m along the aisle), height."""

    track_id: int
    x0: float
    x1: float
    z0: float
    z1: float
    height: float = 1.5
    cls: str = "car"


def parked(track_id: int, side: str, stall: int, length: float = 4.7, width: float = 1.85, depth: float = 0.0, **kw):  # type: ignore[no-untyped-def]
    """A car in stall `stall` of the row on `side`, its aisle-side face `depth` m into the stall."""
    z0 = stall * PITCH + (PITCH - width) / 2
    if side == "right":
        x0, x1 = EDGE + depth, EDGE + depth + length
    else:
        x0, x1 = -EDGE - depth - length, -EDGE - depth
    return Car3D(track_id, x0, x1, z0, z0 + width, **kw)


def row(side: str, stalls: range, empty: set[int] = frozenset(), first_id: int = 1) -> list[Car3D]:  # type: ignore[assignment]
    return [parked(first_id + s, side, s) for s in stalls if s not in empty]


def project(
    car: Car3D, ego_z: float, xs_: tuple[float, float] | None = None
) -> tuple[float, float, float, float] | None:
    """Image box of the car (or of its slice between lateral `xs_`), clipped to the frame."""
    zs = [car.z0 - ego_z, car.z1 - ego_z]
    if min(zs) <= 0.5:
        return None
    x_range = xs_ or (car.x0, car.x1)
    pts = [CAM.project(x, z, up, HORIZON) for x in x_range for z in zs for up in (0.0, car.height)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, y1, x2, y2 = max(min(xs), 0.0), max(min(ys), 0.0), min(max(xs), W - 1.0), min(max(ys), H - 1.0)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


class Lot:
    """Drives down the aisle; feeds the tracker frame by frame."""

    def __init__(self, cars: list[Car3D], cfg: ParkingConfig | None = None, hfov: float = HFOV, seed: int = 0):
        self.cars = cars
        self.tracker = ParkingTracker(cfg or ParkingConfig(), hfov)
        self.rng = np.random.default_rng(seed)
        self.events: list[ParkingEvent] = []
        self.t = 0.0
        self.ego_z = -12.0
        self.extra: list[Track] = []  # people, moving cars... (already projected)
        self.drop: set[int] = set()  # track ids missed this frame

    def frame(self, speed: float, stopped: bool = False, homography: np.ndarray | None = None) -> list[ParkingEvent]:
        boxes = []
        for car in self.cars:
            box = project(car, self.ego_z)
            if box is not None:
                boxes.append((car, box))
        boxes.sort(key=lambda cb: -cb[1][3])  # nearest (lowest bottom edge) first
        tracks = []
        for i, (car, box) in enumerate(boxes):
            # Looking down a row we see each car's aisle-facing end (front or back): judge visibility on that face.
            face = (car.x0, car.x0 + 0.3) if car.x0 > 0 else (car.x1 - 0.3, car.x1)
            face_box = project(car, self.ego_z, face) or box
            hidden = coverage(face_box, [b for _, b in boxes[:i] if b[3] > box[3] + 1])
            if hidden > 0.7 or car.track_id in self.drop:
                continue  # a detector misses mostly hidden cars
            j = self.rng.normal(0.0, 1.0, 4)
            tracks.append(Track(car.track_id, car.cls, 0.9, tuple(float(v) for v in np.add(box, j))))  # type: ignore[arg-type]
        events = self.tracker.update(self.t, W, H, tracks + self.extra, stopped, homography)
        self.events += events
        self.t += 1 / FPS
        self.ego_z += speed / FPS
        return events

    def drive(self, seconds: float, speed: float = 2.5) -> None:
        for _ in range(round(seconds * FPS)):
            self.frame(speed)

    def spots(self) -> list[ParkingEvent]:
        return [e for e in self.events if e.kind == "spot"]


def true_stall_z(stall: int) -> float:
    return stall * PITCH + (PITCH - 1.85) / 2


# ---------------------------------------------------------------------------------------------- spots


def test_one_open_stall_is_found_once_on_the_right_side() -> None:
    lot = Lot(row("right", range(12), empty={6}) + row("left", range(12), first_id=100))
    lot.drive(12.0)
    spots = lot.spots()
    assert len(spots) == 1
    space = spots[0].space
    assert space is not None and space.side == "right" and space.stalls == 1
    assert space.near.track_id == 1 + 5 and space.far.track_id == 1 + 7  # between stalls 5 and 7
    assert space.cars_before >= 1
    # Reported while the space is still ahead of us, at the right distance (its near edge = stall 5's far side).
    ego_at_report = lot.ego_z - 2.5 * (lot.t - spots[0].t) - 2.5 / FPS
    true_z = true_stall_z(5) + 1.85 - ego_at_report
    assert true_z > 4.0 and space.z == pytest.approx(true_z, abs=0.6)
    assert space.visible_depth >= ParkingConfig().min_visible_depth_m


def test_two_adjacent_open_stalls_are_one_spot_of_two() -> None:
    lot = Lot(row("left", range(12), empty={5, 6}) + row("right", range(12), first_id=100))
    lot.drive(12.0)
    spots = lot.spots()
    assert len(spots) == 1 and spots[0].space is not None
    assert spots[0].space.side == "left" and spots[0].space.stalls == 2


def test_full_rows_give_no_spots_despite_jitter() -> None:
    for seed in range(3):
        lot = Lot(row("right", range(14)) + row("left", range(14), first_id=100), seed=seed)
        lot.drive(14.0)
        assert lot.spots() == []


def test_a_detection_dropout_is_not_a_spot() -> None:
    lot = Lot(row("right", range(12)) + row("left", range(12), first_id=100))
    lot.drive(3.0)
    lot.drop = {1 + 6}  # stall 6 missed for 0.4 s (< confirm_s)
    lot.drive(0.4)
    lot.drop = set()
    lot.drive(8.0)
    assert lot.spots() == []


def test_the_end_of_a_row_is_not_a_spot() -> None:
    lot = Lot(row("right", range(6)) + row("left", range(14), first_id=100))
    lot.drive(14.0)
    assert lot.spots() == []


def test_a_small_car_pulled_in_behind_an_suv_is_not_an_open_spot() -> None:
    # Stall 6 holds a small car parked 1 m deep, right behind a tall long SUV in stall 5: from afar the stall looks
    # empty. It must not be reported before we can see into it (and then the small car is seen).
    cars = row("right", range(12), empty={5, 6}) + row("left", range(12), first_id=100)
    cars.append(parked(6, "right", 5, length=5.3, height=1.9, cls="truck"))
    cars.append(parked(7, "right", 6, length=3.6, height=1.4, depth=1.0))
    lot = Lot(cars)
    lot.drive(12.0)
    assert lot.spots() == []


def test_a_person_standing_in_a_stall_fills_it_until_they_leave() -> None:
    lot = Lot(row("right", range(12), empty={6}) + row("left", range(12), first_id=100))
    person_z = true_stall_z(6) + 0.9
    for _ in range(round(6.0 * FPS)):
        z = person_z - lot.ego_z
        x1, y1 = CAM.project(EDGE + 1.0, z, 1.7, HORIZON)
        x2, y2 = CAM.project(EDGE + 1.5, z, 0.0, HORIZON)
        lot.extra = [Track(900, "person", 0.9, (x1, y1, x2, y2))]
        lot.frame(2.5)
    assert lot.spots() == []
    lot.extra = []
    lot.drive(5.0)
    assert len(lot.spots()) == 1


def test_a_car_driving_in_the_aisle_does_not_make_or_fill_spots() -> None:
    lot = Lot(row("right", range(12), empty={6}) + row("left", range(12), first_id=100))
    for _ in range(round(12.0 * FPS)):
        z = 14.0 + 1.0 * lot.t  # a car ahead of us in the aisle, slower than us
        lot.extra = [
            Track(
                950,
                "car",
                0.9,
                project(Car3D(950, -0.9, 0.9, lot.ego_z + z, lot.ego_z + z + 4.5), lot.ego_z) or (0, 0, 1, 1),
            )
        ]
        lot.frame(2.5)
    assert len(lot.spots()) == 1


def test_decisions_do_not_depend_on_the_assumed_field_of_view() -> None:
    # The tracker thinks the camera is 90° wide (it's 60°): every distance is off by the same factor, but spacings
    # are measured in stall pitches, so the same spot is found.
    lot = Lot(row("right", range(12), empty={6}) + row("left", range(12), first_id=100), hfov=90.0)
    lot.drive(12.0)
    spots = lot.spots()
    assert len(spots) == 1 and spots[0].space is not None and spots[0].space.near.track_id == 6


# -------------------------------------------------------------------------------- pulling out, lot, parked


def test_a_car_pulling_out_is_reported_once() -> None:
    cars = row("right", range(12)) + row("left", range(12), first_id=100)
    lot = Lot(cars)
    lot.drive(4.0, speed=1.0)  # crawling, then we wait: stall 8 on the right is ~30 m ahead
    lot.drive(1.0, speed=0.0)
    moving = next(c for c in cars if c.track_id == 1 + 8)
    for k in range(round(3.0 * FPS)):
        backed = 0.8 * k / FPS  # backing out into the aisle at 0.8 m/s
        lot.cars = [c for c in lot.cars if c.track_id != moving.track_id]
        lot.cars.append(replace(moving, x0=moving.x0 - backed, x1=moving.x1 - backed))
        lot.frame(0.0, stopped=True)
    pulls = [e for e in lot.events if e.kind == "pulling_out"]
    assert len(pulls) == 1 and pulls[0].car is not None
    assert pulls[0].car.track_id == 1 + 8 and pulls[0].car.side == "right"


def test_a_lot_is_recognized_when_crawling_past_parked_cars_not_when_driving_past_fast() -> None:
    lot = Lot(row("right", range(14)) + row("left", range(14), first_id=100))
    lot.drive(10.0, speed=2.5)
    assert [e.kind for e in lot.events].count("lot") == 1
    street = Lot(row("right", range(40)) + row("left", range(40), first_id=100))
    street.drive(5.0, speed=13.0)  # ~30 mph past parked cars
    assert "lot" not in [e.kind for e in street.events]


def yaw_homography(degrees: float) -> np.ndarray:
    """Pure rotation about the vertical axis (we turn), as a frame-to-frame homography."""
    a = math.radians(degrees)
    k = np.array([[CAM.f, 0, CAM.cx], [0, CAM.f, HORIZON], [0, 0, 1.0]])
    r = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return k @ r @ np.linalg.inv(k)


def test_parked_after_turning_into_a_stall_and_stopping() -> None:
    lot = Lot([])
    for _ in range(round(4.0 * FPS)):  # turn 80° over 4 s while creeping in
        lot.frame(1.0, homography=yaw_homography(80.0 / (4.0 * FPS)))
    for _ in range(round(11.0 * FPS)):
        lot.frame(0.0, stopped=True)
    assert [e.kind for e in lot.events].count("parked") == 1
    waiting = Lot([])
    for _ in range(round(20.0 * FPS)):  # stopped straight (a stop sign, a pedestrian): not parked yet
        waiting.frame(0.0, stopped=True)
    assert "parked" not in [e.kind for e in waiting.events]
