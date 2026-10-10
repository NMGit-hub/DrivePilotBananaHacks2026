"""Safety regressions from the M2 review: things that must NOT make the co-pilot say "you're up" or "it's green".

Scenarios use a pinhole camera (1280x720, 114° horizontal FOV, 1.3 m high) and cars placed in world coordinates,
with detector-like box jitter, so they exercise the real vehicle-motion → approach → queue → event chain.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from core.config import VehicleMotionConfig, load_config
from core.types import Frame
from perception.detector import Track
from perception.ego_motion import NOT_MEASURED, EgoMotion
from perception.pipeline import AsyncListener, PerceptionResult
from state.traffic_light import LightColor, TrafficLightTracker
from state.vehicle_motion import VehicleMotionTracker, homography_usable
from state.world import WorldEvent, WorldState

W, H = 1280, 720
F = 640 / math.tan(math.radians(57))
CX, CY, CAM_H = 640.0, 360.0, 1.3
IMG = np.full((H, W, 3), 100, np.uint8)
SIGN = Track(50, "stop sign", 0.8, (900.0, 200.0, 960.0, 260.0))


def side_box(x_near: float, x_far: float, z: float, height: float = 1.5) -> tuple[float, float, float, float]:
    """A car seen side-on: spans x_near..x_far (m, + = right), at depth z (m), box from ground to roof."""
    xs = sorted(F * x / z + CX for x in (x_near, x_far))
    return xs[0], F * (CAM_H - height) / z + CY, xs[1], F * CAM_H / z + CY


def rear_box(x: float, z: float, width: float = 1.8, height: float = 1.5) -> tuple[float, float, float, float]:
    """A car seen from behind / the front, centered at lateral x, depth z."""
    return side_box(x - width / 2, x + width / 2, z, height)


def ground_h(dz: float) -> np.ndarray:
    """Frame-to-frame homography of the ground plane when we move forward dz meters."""
    k = dz / (F * CAM_H)
    t = np.array([[1, 0, -CX], [0, 1, -CY], [0, 0, 1.0]])
    return np.linalg.inv(t) @ np.array([[1, 0, 0], [0, 1, 0], [0, -k, 1.0]]) @ t


class Sim:
    """Feeds WorldState one frame at a time: tracks in world coordinates, our own motion, box jitter."""

    def __init__(self, seed: int = 0, jitter_px: float = 1.0) -> None:
        self.events: list[WorldEvent] = []
        self.world = WorldState(load_config(), on_event=self.events.append)
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter_px
        self.i = 0
        self.ego_z = 0.0  # how far we have driven forward (m)
        self.stopped_since: float | None = None

    def step(
        self,
        t: float,
        cars: dict[int, tuple[float, float, float, float]],
        *,
        sign: bool,
        ego_stopped: bool,
        ego_dz: float = 0.0,
        homography: str = "ground",
    ) -> None:
        """One frame. `cars`: track id -> box in pixels (before jitter). `ego_dz`: our motion this frame."""
        self.ego_z += ego_dz
        if ego_stopped and self.stopped_since is None:
            self.stopped_since = t
        if not ego_stopped:
            self.stopped_since = None
        tracks = [SIGN] if sign else []
        for track_id, (x1, y1, x2, y2) in cars.items():
            j = self.rng.normal(0.0, self.jitter, 4)
            tracks.append(Track(track_id, "car", 0.9, (x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3])))
        hom = None
        if ego_dz > 0 and homography == "ground":
            hom = ground_h(ego_dz)
        ego = EgoMotion(
            t=t,
            short=NOT_MEASURED,
            long=NOT_MEASURED,
            value=0.0 if ego_stopped else 5.0,
            stopped=ego_stopped,
            stopped_since=self.stopped_since,
            confident=True,
            homography=hom,
            elapsed_ms=0.0,
        )
        frame = Frame(image=IMG, t=t, index=self.i, captured_at=0.0)
        self.world.on_perception(
            PerceptionResult(frame=frame, tracks=tracks, ego=ego, timings_ms={}, new_segment=self.i == 0)
        )
        self.i += 1

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def order(self) -> list[str]:
        snap = self.world.snapshot()
        assert snap is not None
        return ["us" if e.is_ego else e.approach.value for e in snap.queue.order]


def frames(start: float, end: float) -> list[float]:
    return [round(start + k / 30, 4) for k in range(round((end - start) * 30))]


def left_car(z: float = 8.0, shift: float = 0.0) -> tuple[float, float, float, float]:
    return side_box(-9.0 + shift, -4.5 + shift, z)


# ------------------------------------------------------------------------------ cars that only creep


@pytest.mark.parametrize("seed", range(5))
def test_waiting_car_creeping_to_the_line_is_not_a_departure(seed: int) -> None:
    sim = Sim(seed)
    for t in frames(0.0, 9.0):
        creep = min(max(t - 5.0, 0.0), 1.5)  # 1.5 m forward over 1.5 s (≈ one box height), then stops again
        sim.step(t, {3: left_car(shift=creep)}, sign=t < 2.5, ego_stopped=t >= 2.0)
    assert "vehicle_departed" not in sim.kinds() and "your_turn" not in sim.kinds()
    assert sim.order() == ["left", "us"]
    # Then it really goes: accelerates across the intersection.
    for t in frames(9.0, 12.0):
        drive = 1.5 + 2.0 * (t - 9.0) ** 2  # 2 m/s² from rest
        sim.step(t, {3: left_car(shift=drive)} if drive < 12 else {}, sign=False, ego_stopped=True)
    departed = [e for e in sim.events if e.kind == "vehicle_departed"]
    your_turn = [e for e in sim.events if e.kind == "your_turn"]
    assert len(departed) == 1 and 9.5 <= departed[0].t <= 10.8
    assert len(your_turn) == 1 and your_turn[0].t == departed[0].t


@pytest.mark.parametrize("mode", ["flag_held", "ground", "none"])
def test_our_own_creep_does_not_make_a_waiting_car_depart(mode: str) -> None:
    sim = Sim(1)
    for t in frames(0.0, 11.0):
        creeping = 5.0 <= t < 7.0  # we roll 1.6 m forward to see better
        dz = 1.6 / 60 if creeping else 0.0
        stopped = t >= 2.0 and (mode == "flag_held" or not creeping)
        sim.step(
            t,
            {3: left_car(z=8.0 - sim.ego_z - dz)},
            sign=t < 2.5,
            ego_stopped=stopped,
            ego_dz=dz,
            homography="ground" if mode == "ground" else "none",
        )
    assert "vehicle_departed" not in sim.kinds() and "your_turn" not in sim.kinds()
    assert sim.order() == ["left", "us"]


# ------------------------------------------------------------------------- arrival order with jitter


@pytest.mark.parametrize(("box_h", "sigma"), [(100, 1.0), (50, 1.0), (50, 2.0), (30, 2.0)])
def test_a_car_that_never_moves_stays_arrival_unknown_despite_jitter(box_h: int, sigma: float) -> None:
    for seed in range(20):
        rng = np.random.default_rng(seed)
        tracker = VehicleMotionTracker(VehicleMotionConfig())
        states = {}
        for i in range(60):
            j = rng.normal(0, sigma, 4)
            box = (100 + j[0], 300 + j[1], 100 + 2.6 * box_h + j[2], 300 + box_h + j[3])
            states = tracker.update(10.0 + i / 30, [Track(1, "car", 0.9, box)], None, ego_stopped=True)
        assert states[1].stopped and not states[1].arrival_known, seed


@pytest.mark.parametrize("seed", range(10))
def test_car_revealed_after_we_stop_is_put_ahead_of_us(seed: int) -> None:
    sim = Sim(seed)
    for t in frames(0.0, 7.0):  # it was hidden behind a truck until 4 s
        sim.step(t, {3: left_car()} if t >= 4.0 else {}, sign=t < 2.5, ego_stopped=t >= 2.0)
    assert sim.order() == ["left", "us"]


@pytest.mark.parametrize("seed", range(5))
def test_car_arriving_after_us_goes_behind_us(seed: int) -> None:
    sim = Sim(seed)
    for t in frames(0.0, 9.0):
        x = 4.5 + max(0.0, 5.0 - t) * 3.0  # drives in from the right at 3 m/s, stops at the line at 5 s
        sim.step(t, {4: side_box(x, x + 4.5, 8.0)}, sign=t < 2.5, ego_stopped=t >= 2.0)
    assert sim.order() == ["us", "right"]


def test_waiting_car_seen_while_we_approach_is_ahead_of_us() -> None:
    """While we drive, compensation is approximate: the waiting car may seem to move. It must not be dated to
    our own arrival (a tie would hand us the right of way over a car on our left)."""
    sim = Sim(2)
    for t in frames(0.0, 6.0):
        moving = t < 2.0
        dz = 4.0 / 60 if moving else 0.0  # 2 m/s, then stop
        sim.step(t, {3: left_car(z=12.0 - sim.ego_z - dz)}, sign=True, ego_stopped=not moving, ego_dz=dz)
    assert sim.order() == ["left", "us"]


# ----------------------------------------------------------------------------- behind another car


def test_stopping_behind_a_lead_car_waits_until_we_reach_the_line() -> None:
    sim = Sim(3)
    for t in frames(0.0, 11.0):
        lead_z = 6.0 if t < 6.0 else 6.0 + 3.0 * (t - 6.0)  # the car in front of us goes at 6 s
        pull_up = 7.5 <= t < 9.0
        dz = 4.0 / 45 if pull_up else 0.0  # we pull up 4 m to the line, stop at 9 s
        cars = {7: rear_box(0.0, lead_z - sim.ego_z - dz)} if lead_z - sim.ego_z < 25 else {}
        x_right = 4.5 + max(0.0, 4.0 - t) * 3.0  # a car on the right reaches its line at 4 s
        cars[4] = side_box(x_right, x_right + 4.5, 8.0 - sim.ego_z - dz)
        sim.step(t, cars, sign=t < 9.5, ego_stopped=(2.0 <= t < 7.5) or t >= 9.0, ego_dz=dz)
    arrived = [e for e in sim.events if e.kind == "arrived_at_stop"]
    assert len(arrived) == 1 and arrived[0].t >= 9.0  # not at 2 s, behind the other car
    assert sim.order() == ["right", "us"]  # the right car reached its line first
    assert "your_turn" not in sim.kinds()


def test_only_the_front_car_of_an_approach_joins() -> None:
    sim = Sim(4)
    for t in frames(0.0, 6.0):
        cars = {3: left_car(), 8: side_box(-15.0, -10.5, 8.0)}  # a second car queued behind it
        sim.step(t, cars, sign=t < 2.5, ego_stopped=t >= 2.0)
    snap = sim.world.snapshot()
    assert snap is not None and [e.track_id for e in snap.queue.order if not e.is_ego] == [3]


# ------------------------------------------------------------------------------------- lights


def draw_light(image: np.ndarray, x: int, y: int, lit: str, lamp: int) -> tuple[float, float, float, float]:
    import cv2

    w, h = lamp * 2 + 8, lamp * 6 + 16
    cv2.rectangle(image, (x, y), (x + w, y + h), (30, 30, 30), -1)
    hues = {"red": 2, "yellow": 25, "green": 88}
    for i, color in enumerate(("red", "yellow", "green")):
        center = (x + w // 2, y + 6 + lamp + i * (2 * lamp + 2))
        if color == lit:
            rim = cv2.cvtColor(np.uint8([[[hues[color], 215, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.circle(image, center, lamp, rim, -1)
            cv2.circle(image, center, lamp // 2, (245, 250, 255), -1)
        else:
            cv2.circle(image, center, lamp, (45, 45, 45), -1)
    return float(x), float(y), float(x + w), float(y + h)


def test_a_cross_street_light_turning_green_is_not_our_green() -> None:
    tracker = TrafficLightTracker(load_config().traffic_light)
    fired = []
    for i in range(120):
        image = np.full((H, W, 3), (120, 110, 100), np.uint8)
        ours = draw_light(image, 620, 150, "red", lamp=12)  # ours stays red
        cross = draw_light(image, 820, 160, "red" if i < 45 else "green", lamp=7)  # the cross street's goes green
        tracks = [Track(1, "traffic light", 0.7, ours), Track(2, "traffic light", 0.5, cross)]
        reading, green = tracker.update(i / 30, image, tracks, ego_stopped=True)
        fired.append(green)
    assert not any(fired)
    assert reading.color is LightColor.RED and reading.confident


# --------------------------------------------------------------------------------- robustness


@pytest.mark.parametrize(
    "h_mat",
    [
        np.full((3, 3), np.nan),
        np.array([[1.0, 0, 5000.0], [0, 1.0, 0], [0, 0, 1.0]]),  # absurd jump
        np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 1.0]]),  # absurd zoom
        np.zeros((3, 3)),
    ],
)
def test_bad_homographies_are_ignored(h_mat: np.ndarray) -> None:
    assert not homography_usable(h_mat, 200.0, 0.15)
    tracker = VehicleMotionTracker(VehicleMotionConfig())
    for i in range(20):
        states = tracker.update(i / 30, [Track(1, "car", 0.9, (100.0, 300.0, 360.0, 400.0))], h_mat, False)
    assert 1 in states


def test_homography_with_nothing_stored_yet() -> None:
    """Regression (found on a real clip): remapping with empty histories crashed (OpenCV returns None)."""
    tracker = VehicleMotionTracker(VehicleMotionConfig())
    tracker.update(0.0, [Track(1, "car", 0.9, (100.0, 300.0, 360.0, 400.0))], None, False)
    tracker.update(0.1, [], None, False)  # we moved without a homography: histories cleared
    tracker.update(0.2, [], ground_h(0.1), False)  # must not raise


def test_vehicle_motion_is_cheap_per_frame() -> None:
    tracker = VehicleMotionTracker(VehicleMotionConfig())
    h_mat = ground_h(0.05)
    times = []
    for i in range(90):
        tracks = [Track(k, "car", 0.9, (50.0 + 45 * k, 400.0, 200.0 + 45 * k, 470.0)) for k in range(25)]
        start = time.perf_counter()
        tracker.update(i / 30, tracks, h_mat, False)
        times.append(time.perf_counter() - start)
    assert np.median(times[30:]) < 0.015  # 25 cars; ~2-5 ms typical (generous bound: CI machines vary)


# ------------------------------------------------------------------------------ async listener


def result(i: int, new_segment: bool = False) -> PerceptionResult:
    ego = EgoMotion(
        t=i / 30,
        short=NOT_MEASURED,
        long=NOT_MEASURED,
        value=0.0,
        stopped=True,
        stopped_since=0.0,
        confident=True,
        homography=None,
        elapsed_ms=0.0,
    )
    return PerceptionResult(
        frame=Frame(image=IMG, t=i / 30, index=i, captured_at=0.0),
        tracks=[],
        ego=ego,
        timings_ms={},
        new_segment=new_segment,
    )


def test_async_listener_drops_the_oldest_but_never_a_seek() -> None:
    gate = threading.Event()
    seen: list[tuple[int, bool]] = []

    def slow(r: PerceptionResult) -> None:
        gate.wait(5.0)
        seen.append((r.frame.index, r.new_segment))

    listener = AsyncListener(slow, "test", max_pending=2).start()
    listener(result(0))
    time.sleep(0.05)  # the consumer is now blocked on frame 0
    listener(result(1, new_segment=True))
    listener(result(2))
    listener(result(3))  # frame 1 (the seek) is dropped: its flag moves to frame 3
    gate.set()
    listener.stop()
    assert seen == [(0, False), (2, False), (3, True)] and listener.dropped == 1


def test_async_listener_survives_exceptions() -> None:
    calls = []

    def flaky(r: PerceptionResult) -> None:
        calls.append(r.frame.index)
        if r.frame.index == 0:
            raise RuntimeError("boom")

    listener = AsyncListener(flaky, "flaky").start()
    listener(result(0))
    listener(result(1))
    listener.stop()
    assert calls == [0, 1] and listener.handled == 2
